# Local model artifacts

The local training workflow writes these files here:

```text
models/emotion_cnn_model.keras
models/emotion_cnn_model.metadata.json
models/model_selection.json
models/model_validation.json
```

Keras/HDF5 model files are intentionally ignored by Git. They are binary build
artifacts rather than source code. The JSON files are safe to commit: the
metadata records how the model was trained, the selection report records the
subject-grouped cross-validation comparison, and the validation report records
an independently reproduced evaluation of the saved artifact.

The released model and metadata are also stored under an immutable logical
version in the private S3 artifact bucket. The deployed baseline embeds the
same model in its Docker image, making an image tag identify both code and
weights. MLflow-based model promotion is intentionally left for a later phase.
