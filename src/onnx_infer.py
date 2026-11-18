# src/onnx_infer.py
"""ONNX inference without torch: tokenizers + onnxruntime. Used by serve_onnx.py and onnx_pyfunc.py.

A model directory holds model.onnx, tokenizer.json and meta.json (labels, max_length).
"""
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class OnnxClassifier:
    def __init__(self, model_dir: str | Path):
        d = Path(model_dir)
        meta = json.loads((d / "meta.json").read_text())
        self.labels: list[str] = meta["labels"]
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(max_length=int(meta["max_length"]))
        self.tok.no_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "2"))  # match the pod's CPU limit
        self.sess = ort.InferenceSession(str(d / "model.onnx"), opts, providers=["CPUExecutionProvider"])

    def scores(self, text: str) -> dict[str, float]:
        e = self.tok.encode(text)
        feed = {"input_ids": np.array([e.ids], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids], dtype=np.int64)}
        logits = self.sess.run(["logits"], feed)[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return {label: float(s) for label, s in zip(self.labels, p)}
