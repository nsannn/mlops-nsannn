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

## Finding: CPU throttling, fixed before the final measurement

The first run with the default settings had p50 8.9 ms but p95 93 ms with 1 client (22 req/s). The pod's
`cpu.stat` showed 129 throttling events (78 s in total): torch started 10 threads (one per core it could see)
inside a 2-CPU limit, burned the quota in bursts and was paused until the next 100 ms CFS period.

| Change | 1 client p95 / req/s | 4 clients p95 / req/s |
| --- | --- | --- |
| Defaults (10 torch threads) | 92 ms / 23 | 85 ms / 71 |
| `OMP_NUM_THREADS=2` in `k8s/api.yaml` | 6.3 ms / 180 | 103 ms / 50 |
| + one inference at a time (`INFERENCE_LOCK` in `src/serve.py`) | **7.2 ms / 161** | **71 ms / 100** |

Matching the thread count to `limits.cpu` removed the stalls for single requests. Parallel requests still
oversubscribed the 2 CPUs (4 requests × 2 threads), so inference is serialized: each call uses both cores and
the others queue. With 4 clients the latency is mostly that queueing; more throughput needs more replicas
(HPA, U2-E3), not more threads per pod.
