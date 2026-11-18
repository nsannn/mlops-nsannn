# ONNX variant vs PyTorch (U2-E4)

Date: 2026-11-18 · Source model: `tweet-emotion` v2 (`@champion`) · ONNX model: `tweet-emotion-onnx` v1
(`@champion`, INT8, run `04bf43a1`) · Export: [`scripts/export_onnx.py`](../scripts/export_onnx.py)

The PyTorch model was exported with `torch.onnx.export` (opset 17, dynamic batch and sequence length) and quantized
with ONNX Runtime dynamic INT8 quantization. The export refuses to register unless the ONNX probabilities match
PyTorch (max difference 7.2e-7 on 50 test tweets), F1 passes the quality gate and stays within 0.005 of the source.

Both APIs ran side by side in the kind cluster with the same limits (2 CPU, 2 threads, one inference at a time) and
the same test: `scripts/load_test.py`, 100 requests through `kubectl port-forward`, 6 warm-up requests not counted,
second of two runs.

| | PyTorch API (`emotion-api`) | ONNX API (`emotion-api-onnx`) | Change |
| --- | --- | --- | --- |
| Image | 1 940 MB (`api:sha-f325e5c`) | **415 MB** (`api-onnx:sha-c67c87d`) | 4.7× smaller |
| Main packages | torch, transformers, mlflow | onnxruntime, tokenizers, mlflow-skinny | no torch |
| Model file | 42.6 MiB (fp32) | **10.8 MiB** (INT8) | 3.9× smaller |
| Pod memory after the test | 1 010 MiB | **140 MiB** | 7.2× less |
| Ready after a rollout | 10–11 s | 6 s | |
| 1 client: p50 / p95 / p99 | 5.5 / 6.6 / 7.4 ms | **3.2 / 4.0 / 4.7 ms** | p95 1.7× lower |
| 1 client: throughput | 175.6 req/s | **296.7 req/s** | 1.7× |
| 4 clients: p50 / p95 / p99 | 26.1 / 74.3 / 77.6 ms | **6.0 / 9.0 / 20.9 ms** | p95 8× lower |
| 4 clients: throughput | 107.6 req/s | **586.6 req/s** | 5.4× |
| CPU throttling events | 19 | 2 | |
| Test accuracy / macro F1 | 0.880 / 0.8464 | 0.880 / **0.8461** | −0.0003 F1 |

Model-only inference on the laptop (2 threads, no HTTP), measured during the export check: PyTorch p50/p95
2.63 / 3.56 ms, ONNX fp32 1.30 / 2.08 ms, ONNX INT8 0.72 / 1.11 ms. End to end, HTTP and the port-forward add
about 3 ms to every request, so the API-level gain at 1 client is smaller than the model-level gain.

## Trade-off

The INT8 ONNX model gives the same predictions on all but a handful of test tweets (F1 −0.0003) while the image is
4.7× smaller, memory 7× lower and throughput under load 5.4× higher. The gain under load is larger than at one
client because each ONNX inference is short enough that requests queue far less behind the inference lock.
The costs are an extra export step, a second registered model to keep in sync with `tweet-emotion@champion`, and a
quantized model that is not bit-identical to the trained one (checked automatically by the export).
The PyTorch API stays the reference; switching production would mean pointing `emotion-api` at the ONNX image and
`MODEL_URI`, or shifting traffic gradually (canary, U2-E5).
