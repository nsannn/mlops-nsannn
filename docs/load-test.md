# Load test: inference API

Date: 2026-11-16 · Model: `tweet-emotion` v1 (`@champion`, run `52fdd3aa`) ·
Image: `ghcr.io/nsannn/mlops-nsannn/api:local-6813243-lock` (built locally from `docker/Dockerfile.api`;
CI could not publish while the GitHub account was locked for billing)

Setup: kind single node on a laptop (WSL2, 20 logical CPUs), 1 replica, limits 2 CPU / 2 GiB,
`OMP_NUM_THREADS=2`. Requests go through `kubectl port-forward`, which adds tunnel overhead, so the numbers
are pessimistic. 6 warm-up requests are not counted. Each row is the second of two runs.
Script: `scripts/load_test.py`.

| Clients | Requests | Errors | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 100 | 0 | 6.0 | **7.2** | 7.9 | 160.8 |
| 4 | 100 | 0 | 31.3 | 71.0 | 75.1 | **100.3** |

| Size | Value |
| --- | --- |
| Model weights (`model.safetensors`) | 42.6 MiB (44.7 MB) |
| API image | 1 940 MB (CPU-only torch dominates; no model inside) |
| Ready after a rollout (model download included) | 10–11 s |
