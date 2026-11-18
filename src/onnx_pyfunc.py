# src/onnx_pyfunc.py
"""MLflow pyfunc for the ONNX emotion model, logged as code by scripts/export_onnx.py."""
from pathlib import Path

import mlflow

from onnx_infer import OnnxClassifier


class OnnxEmotion(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.clf = OnnxClassifier(Path(context.artifacts["model.onnx"]).parent)

    def predict(self, context, model_input, params=None):
        texts = model_input.iloc[:, 0].tolist() if hasattr(model_input, "iloc") else list(model_input)
        labels = []
        for text in texts:
            scores = self.clf.scores(text)
            labels.append(max(scores, key=scores.get))
        return labels


mlflow.models.set_model(OnnxEmotion())
