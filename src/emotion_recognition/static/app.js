"use strict";

const camera = document.querySelector("#camera");
const imagePreview = document.querySelector("#image-preview");
const imageInput = document.querySelector("#image-input");
const overlay = document.querySelector("#overlay");
const capture = document.querySelector("#capture");
const placeholder = document.querySelector("#camera-placeholder");
const cameraButton = document.querySelector("#camera-button");
const analyzeButton = document.querySelector("#analyze-button");
const liveButton = document.querySelector("#live-button");
const statusText = document.querySelector("#status");
const emotionText = document.querySelector("#emotion");
const confidenceText = document.querySelector("#confidence");
const inferenceText = document.querySelector("#inference-time");
const faceCountText = document.querySelector("#face-count");
const confidenceBar = document.querySelector("#confidence-bar");

let stream = null;
let liveTimer = null;
let requestInFlight = false;
let previewUrl = null;

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.classList.toggle("error", isError);
}

function clearOverlay() {
  overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
}

function resetResult() {
  emotionText.textContent = "—";
  confidenceText.textContent = "—";
  inferenceText.textContent = "—";
  faceCountText.textContent = "—";
  confidenceBar.style.width = "0";
  clearOverlay();
}

function clearUploadedImage() {
  if (previewUrl !== null) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
  imagePreview.removeAttribute("src");
  imagePreview.hidden = true;
  camera.hidden = false;
  document.querySelector("#camera-stage").style.removeProperty("aspect-ratio");
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("Camera access requires an HTTPS page or localhost.", true);
    return;
  }

  try {
    clearUploadedImage();
    resetResult();
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });
    camera.srcObject = stream;
    await camera.play();

    overlay.width = camera.videoWidth;
    overlay.height = camera.videoHeight;
    capture.width = camera.videoWidth;
    capture.height = camera.videoHeight;

    placeholder.hidden = true;
    cameraButton.textContent = "Stop camera";
    analyzeButton.disabled = false;
    liveButton.disabled = false;
    setStatus("Camera ready. Analyze one frame or start live mode.");
  } catch (error) {
    setStatus(`Camera unavailable: ${error.message}`, true);
  }
}

function stopLiveMode() {
  if (liveTimer !== null) {
    window.clearInterval(liveTimer);
    liveTimer = null;
  }
  liveButton.textContent = "Start live mode";
}

function stopCamera() {
  stopLiveMode();
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  camera.srcObject = null;
  placeholder.hidden = false;
  cameraButton.textContent = "Start camera";
  analyzeButton.disabled = true;
  liveButton.disabled = true;
  requestInFlight = false;
  resetResult();
  setStatus("Camera stopped.");
}

function canvasBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Frame capture failed"))),
      "image/jpeg",
      0.9,
    );
  });
}

function drawDetection(face, label, confidence, mirrored) {
  const context = overlay.getContext("2d");
  context.clearRect(0, 0, overlay.width, overlay.height);

  const boxX = mirrored ? overlay.width - face.x - face.width : face.x;
  context.strokeStyle = "#5eead4";
  context.lineWidth = Math.max(3, overlay.width / 260);
  context.strokeRect(boxX, face.y, face.width, face.height);

  const caption = `${label} ${(confidence * 100).toFixed(1)}%`;
  context.font = `700 ${Math.max(18, overlay.width / 34)}px system-ui`;
  const captionWidth = context.measureText(caption).width + 20;
  const captionHeight = Math.max(34, overlay.height / 14);
  const captionY = Math.max(0, face.y - captionHeight);
  context.fillStyle = "#5eead4";
  context.fillRect(boxX, captionY, captionWidth, captionHeight);
  context.fillStyle = "#06201d";
  context.fillText(caption, boxX + 10, captionY + captionHeight * 0.72);
}

function renderPrediction(result, mirrored = true) {
  faceCountText.textContent = String(result.face_count);
  inferenceText.textContent = `${result.inference_time_ms.toFixed(0)} ms`;

  if (result.face_count === 0) {
    emotionText.textContent = "No face";
    confidenceText.textContent = "—";
    confidenceBar.style.width = "0";
    clearOverlay();
    setStatus("No face detected. Try a clear, front-facing photo with good lighting.");
    return;
  }

  const detection = result.faces[0];
  const prediction = detection.prediction;
  const confidencePercent = prediction.confidence * 100;
  emotionText.textContent = prediction.label;
  confidenceText.textContent = `${confidencePercent.toFixed(1)}%`;
  confidenceBar.style.width = `${confidencePercent}%`;
  drawDetection(
    detection.face,
    prediction.label,
    prediction.confidence,
    mirrored,
  );
  setStatus("Prediction complete. Frames are processed in memory and not stored.");
}

async function requestPrediction(file) {
  const form = new FormData();
  form.append("file", file, file.name || "webcam-frame.jpg");

  const response = await fetch("/v1/predictions?largest_only=true", {
    method: "POST",
    body: form,
  });
  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.detail || `Prediction failed (${response.status})`);
  }
  return result;
}

async function analyzeFrame() {
  if (!stream || requestInFlight || camera.readyState < 2) {
    return;
  }

  requestInFlight = true;
  analyzeButton.disabled = true;
  setStatus("Analyzing frame…");

  try {
    capture.getContext("2d").drawImage(camera, 0, 0, capture.width, capture.height);
    const blob = await canvasBlob(capture);
    const result = await requestPrediction(
      new File([blob], "webcam-frame.jpg", { type: "image/jpeg" }),
    );
    renderPrediction(result, true);
  } catch (error) {
    stopLiveMode();
    setStatus(error.message, true);
  } finally {
    requestInFlight = false;
    analyzeButton.disabled = !stream;
  }
}

async function analyzeUploadedImage(file) {
  if (requestInFlight) {
    return;
  }

  requestInFlight = true;
  imageInput.disabled = true;
  resetResult();
  setStatus("Analyzing uploaded image…");

  try {
    const result = await requestPrediction(file);
    renderPrediction(result, false);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    requestInFlight = false;
    imageInput.disabled = false;
    imageInput.value = "";
  }
}

function showUploadedImage(file) {
  if (!file) {
    return;
  }

  if (stream) {
    stopCamera();
  } else {
    stopLiveMode();
  }
  clearUploadedImage();
  previewUrl = URL.createObjectURL(file);
  imagePreview.src = previewUrl;
  imagePreview.onload = () => {
    camera.hidden = true;
    imagePreview.hidden = false;
    placeholder.hidden = true;
    overlay.width = imagePreview.naturalWidth;
    overlay.height = imagePreview.naturalHeight;
    document.querySelector("#camera-stage").style.aspectRatio =
      `${imagePreview.naturalWidth} / ${imagePreview.naturalHeight}`;
    analyzeUploadedImage(file);
  };
  imagePreview.onerror = () => {
    clearUploadedImage();
    placeholder.hidden = false;
    setStatus("The selected file could not be previewed.", true);
  };
}

function toggleLiveMode() {
  if (liveTimer !== null) {
    stopLiveMode();
    setStatus("Live mode stopped.");
    return;
  }
  liveButton.textContent = "Stop live mode";
  analyzeFrame();
  liveTimer = window.setInterval(analyzeFrame, 1250);
  setStatus("Live mode active. Processing approximately one frame per second.");
}

cameraButton.addEventListener("click", () => {
  if (stream) {
    stopCamera();
  } else {
    startCamera();
  }
});
analyzeButton.addEventListener("click", analyzeFrame);
liveButton.addEventListener("click", toggleLiveMode);
imageInput.addEventListener("change", () => {
  showUploadedImage(imageInput.files?.[0]);
});
window.addEventListener("pagehide", () => {
  stopCamera();
  clearUploadedImage();
});
