# scripts/export_onnx.py
"""U2-E4: export the @champion PyTorch model to ONNX, quantize to INT8, check it, register it.

  1. load models:/tweet-emotion@champion (transformers components)
  2. torch.onnx.export -> fp32; onnxruntime quantize_dynamic -> int8
  3. parity: max |torch - onnx| probability difference on 50 test tweets
  4. accuracy / macro F1 of both variants on the test split
  5. refuse unless parity < 1e-4, F1 >= the gate and within 0.005 of the source model
  6. log as an MLflow pyfunc (model.onnx + tokenizer.json + meta.json), register tweet-emotion-onnx, alias @champion

Run: python scripts/export_onnx.py [--variant int8|fp32] [--no-register]   (needs MLFLOW_TRACKING_URI)
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from onnx_infer import OnnxClassifier
from train import EXPERIMENT, git_commit

SOURCE, TARGET = "tweet-emotion", "tweet-emotion-onnx"
INPUTS = ["input_ids", "attention_mask", "token_type_ids"]


def evaluate(clf: OnnxClassifier, test: pd.DataFrame) -> dict:
    preds = []
    for text in test["text"]:
        scores = clf.scores(text)
        preds.append(clf.labels.index(max(scores, key=scores.get)))
    return {"accuracy": accuracy_score(test["label"], preds), "f1": f1_score(test["label"], preds, average="macro")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["int8", "fp32"], default="int8")
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    threshold = yaml.safe_load((ROOT / "params.yaml").read_text())["gate"]["threshold"]
    client = mlflow.MlflowClient()
    src = client.get_model_version_by_alias(SOURCE, "champion")
    src_run = client.get_run(src.run_id)
    comp = mlflow.transformers.load_model(f"models:/{SOURCE}/{src.version}", return_type="components")
    model, tok = comp["model"].eval(), comp["tokenizer"]
    max_length = int(src_run.data.params["max_length"])
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    test = pd.read_parquet(ROOT / "data/processed/test.parquet")

    work = Path(tempfile.mkdtemp())
    tok.save_pretrained(work / "tok")
    sample = tok(["i am so happy today", "ok"], padding=True, return_tensors="pt")
    torch.onnx.export(model, tuple(sample[n] for n in INPUTS), work / "model.fp32.onnx",
                      input_names=INPUTS, output_names=["logits"],
                      dynamic_axes={n: {0: "batch", 1: "seq"} for n in INPUTS} | {"logits": {0: "batch"}},
                      opset_version=17, dynamo=False)  # TorchScript exporter: no onnxscript needed
    quantize_dynamic(str(work / "model.fp32.onnx"), str(work / "model.int8.onnx"), weight_type=QuantType.QInt8)

    results = {}
    for variant in ("fp32", "int8"):
        d = work / variant
        d.mkdir()
        shutil.copy(work / f"model.{variant}.onnx", d / "model.onnx")
        shutil.copy(work / "tok" / "tokenizer.json", d / "tokenizer.json")
        (d / "meta.json").write_text(json.dumps({"labels": labels, "max_length": max_length,
                                                 "source": f"{SOURCE}/{src.version}", "variant": variant}))
        results[variant] = {**evaluate(OnnxClassifier(d), test),
                            "size_mib": (d / "model.onnx").stat().st_size / 2**20}

    clf32 = OnnxClassifier(work / "fp32")
    diffs = []
    with torch.no_grad():
        for text in test["text"][:50]:
            e = tok(text, truncation=True, max_length=max_length, return_tensors="pt")
            p_torch = torch.softmax(model(**e).logits[0], -1).numpy()
            diffs.append(np.abs(p_torch - np.array(list(clf32.scores(text).values()))).max())
    parity = float(max(diffs))

    source_f1 = float(src_run.data.metrics["f1"])
    chosen = results[args.variant]
    print(json.dumps({"source": f"{SOURCE} v{src.version}", "source_f1": source_f1,
                      "parity_max_prob_diff": parity, **results}, indent=2))
    if parity > 1e-4 or chosen["f1"] < threshold or source_f1 - chosen["f1"] > 0.005:
        sys.exit(f"refusing to register: parity {parity:.1e}, {args.variant} F1 {chosen['f1']:.4f} "
                 f"(gate {threshold}, source {source_f1:.4f})")
    if args.no_register:
        return

    mlflow.set_experiment(EXPERIMENT)
    d = work / args.variant
    with mlflow.start_run(run_name=f"onnx-{args.variant}-from-v{src.version}"):
        mlflow.set_tags({"git_commit": git_commit(), "source_model": f"{SOURCE}/{src.version}",
                         "variant": args.variant, "runner": "laptop"})
        mlflow.log_metrics({"parity_max_prob_diff": parity,
                            **{f"{v}_{k}": m for v, r in results.items() for k, m in r.items()}})
        info = mlflow.pyfunc.log_model(name="model", python_model=str(ROOT / "src/onnx_pyfunc.py"),
                                       code_paths=[str(ROOT / "src/onnx_infer.py")],
                                       artifacts={n: str(d / n) for n in ("model.onnx", "tokenizer.json", "meta.json")},
                                       pip_requirements=str(ROOT / "requirements-api-onnx.txt"))
    mv = mlflow.register_model(info.model_uri, TARGET)
    client.set_registered_model_alias(TARGET, "champion", mv.version)
    print(f"{TARGET} v{mv.version} ({args.variant}, from {SOURCE} v{src.version}) -> @champion")


if __name__ == "__main__":
    main()
