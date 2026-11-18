# MLOps U1 + U2 — Tweet Emotion Detection

This project builds a reproducible training pipeline for a model that classifies tweets into 6 emotions
(sadness, joy, love, anger, fear, surprise). It uses Git, DVC, Docker and a local Kubernetes (`kind`) cluster,
with every experiment tracked in an MLflow server that runs inside the cluster. U2 adds hyperparameter search,
a Model Registry with `@champion`, a training Job and an inference API in the same cluster, and CI/CD with a
quality gate and rollback.

- **Model:** [`google/bert_uncased_L-4_H-256_A-4`](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4) (11M parameters), fine-tuned on CPU
- **Data:** [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion), seeded 10k-tweet stratified sample
- **Submission:** tags `u1-final` and `u2-final` · requirements in [`docs/requirements.md`](docs/requirements.md) (v1.6) ·
  reports [`docs/u1-report.md`](docs/u1-report.md), [`docs/u2-report.md`](docs/u2-report.md)

---

## 1. Running the project from zero

### Folder structure

The project follows the required structure: `src/`, `docker/`, `k8s/`, `docs/`.

| Path | Contents |
| --- | --- |
| `src/` | Pipeline code: `prepare_data.py`, `train.py`, `evaluate.py`, `model_utils.py`; inference API `serve.py`; ONNX variant `serve_onnx.py`, `onnx_infer.py`, `onnx_pyfunc.py` |
| `scripts/` | `fetch_data.py` (pinned dataset sample), `hpo.py` (Optuna), `registry.py` (aliases), `check_model_gate.py`, `deploy.sh`, `load_test.py`, `export_onnx.py` |
| `docker/` | `Dockerfile` (training image), `Dockerfile.mlflow` (MLflow server image), `Dockerfile.api` (inference image), `Dockerfile.api-onnx` (slim ONNX inference image) |
| `k8s/` | `kind-config.yaml` (cluster as code), MLflow + PostgreSQL, `train-job.yaml`, `api.yaml`, `api-onnx.yaml` |
| `tests/` | Unit tests for the quality gate and both API contracts |
| `.github/workflows/` | `ml-pipeline.yml`: lint → test → model-gate → build-push |
| `docs/` | Requirements, reports, CI/CD evidence, load test, image size, security scan, SBOM |
| `params.yaml`, `dvc.yaml`, `dvc.lock` | Hyperparameters and quality gate, pipeline definition, pipeline lock (data and output hashes) |
| `reproduce.py` | Re-creates any historical MLflow run |

### Prerequisites

WSL2 Ubuntu with Docker Desktop (WSL integration on), plus `uv`, `kind`, `kubectl` and `git`.
All commands below run in the Ubuntu terminal.

### Virtual environment

```bash
git clone https://github.com/nsannn/mlops-nsannn.git && cd mlops-nsannn
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip sync --index-strategy unsafe-best-match requirements.txt   # pinned versions, CPU-only torch
```

### Data and pipeline (DVC)

```bash
# Data: on the author's machine the DVC remote is ~/dvcstore
dvc pull
# Elsewhere: rebuild the identical sample from Hugging Face, then check it matches the pinned hash
python scripts/fetch_data.py && dvc status

# Pipeline: prepare -> train -> evaluate (needs MLFLOW_TRACKING_URI, see section 2)
dvc repro            # right after dvc pull every stage is skipped; use --force to train again
dvc dag              # shows the pipeline graph
```

Changing a value under `train:` in `params.yaml` and running `dvc repro` re-runs only `train` and `evaluate`.

---

## 2. Kubernetes infrastructure and MLflow (U1-R4)

The infrastructure runs in its own `mlops` namespace: PostgreSQL stores run metadata, the MLflow server
stores artifacts and models, and each has its own PVC, so data survives pod restarts.

### Running in the cluster (kind)

```bash
# 1. Cluster from code
kind create cluster --config k8s/kind-config.yaml

# 2. Namespace and the database secret (password generated, never committed)
kubectl apply -f k8s/namespace.yaml
kubectl create secret generic mlflow-db -n mlops \
  --from-literal=POSTGRES_USER=mlflow --from-literal=POSTGRES_DB=mlflow \
  --from-literal=POSTGRES_PASSWORD="$(openssl rand -hex 16)"

# 3. MLflow server image (tag = the commit that added its Dockerfile), loaded into the cluster
MLFLOW_IMAGE=$(grep -o 'mlflow-server:sha-[0-9a-f]*' k8s/mlflow.yaml)
docker build -f docker/Dockerfile.mlflow -t "$MLFLOW_IMAGE" docker/
kind load docker-image "$MLFLOW_IMAGE" --name mlops

# 4. PostgreSQL + MLflow
kubectl apply -f k8s/postgres.yaml -f k8s/mlflow.yaml
kubectl wait --for=condition=Available deploy --all -n mlops --timeout=300s
curl http://localhost:5000/health          # OK
```

### Training into the cluster MLflow

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
dvc repro --force
```

### Ports, logins and where to see the results

| What | Address | Login |
| --- | --- | --- |
| MLflow UI and API | [http://localhost:5000](http://localhost:5000) (kind maps host port 5000 → NodePort 30500 → Service `mlflow`) | none |
| MLflow inside the cluster | `http://mlflow.mlops.svc.cluster.local:5000` | none |
| PostgreSQL | `postgres.mlops.svc:5432`, cluster-internal only | user `mlflow`; password: `kubectl get secret mlflow-db -n mlops -o jsonpath='{.data.POSTGRES_PASSWORD}' \| base64 -d` |
| Alternative access | `kubectl port-forward -n mlops svc/mlflow 5000:5000` | none |

* **MLflow UI:** open [http://localhost:5000](http://localhost:5000), switch the toggle at the top left from
  **GenAI** to **Model training**, then open the experiment **`tweet-emotion`**. Each run shows its params,
  metrics (accuracy, macro F1, `eval/*`), `git_commit` and `dvc_raw_md5` tags, dataset and logged model.
* **Latest metrics:** [`metrics.json`](metrics.json).
* **Where artifacts physically live:** PVC `mlflow-artifacts`, a `local-path` volume under
  `/var/local-path-provisioner/` inside the `mlops-control-plane` container. They survive pod restarts,
  but not `kind delete cluster`.

### Experiment results

| Run | Change | Accuracy | Macro F1 | Training run |
| --- | --- | --- | --- | --- |
| exp 1 | baseline: lr 1e-4, 2 epochs, max_length 32 | **0.874** | **0.840** | 1.1 min |
| exp 2 | learning rate 3e-4 | 0.858 | 0.821 | 1.0 min |
| exp 3 (default) | max_length 24 | 0.860 | 0.834 | 52 s |

Exp 3 was the U1 default: it met the then ≤ 60 s training target (NFR-04) at a small accuracy cost. U2 changed the
defaults after the hyperparameter search (section 6).

---

## 3. Docker image and security (U1-R3)

The training image is a multi-stage build that runs as a non-root user (`trainer`, `uid=1000`). It contains
pinned dependencies, the code, the prepared data and the pinned model weights, so it trains fully offline.

* **GHCR image:** `ghcr.io/nsannn/mlops-nsannn/train:sha-af2c8df` (tag = Git commit, never `latest`)
* **Size and scan:** 2 070 MB (Docker `.Size`); Trivy found **0 CRITICAL**, 46 HIGH reviewed. Details are in
  [`docs/image-size.md`](docs/image-size.md), [`docs/security-scan.md`](docs/security-scan.md),
  [`docs/trivy-train.json`](docs/trivy-train.json) and the SBOM [`docs/sbom-train.json`](docs/sbom-train.json).

```bash
docker run --rm -e MLFLOW_TRACKING_URI=sqlite:////tmp/m.db ghcr.io/nsannn/mlops-nsannn/train:sha-af2c8df
```

---

## 4. Reproducing an experiment (U1-R2)

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
time python reproduce.py <run_id>
```

The script reads the run from MLflow, checks out its Git commit and the same DVC data, restores its params,
re-runs the pipeline, and compares accuracy and F1 (tolerance 0.005). The new run is tagged `reproduces=<run_id>`,
and the repo is restored to the starting branch. Measured: about 85 s with Δ = 0.

---

## 5. Before every session

Start Docker Desktop first; the kind cluster and MLflow come back on their own (about 30 s). Then:

```bash
cd ~/mlops-nsannn && source .venv/bin/activate && export MLFLOW_TRACKING_URI=http://localhost:5000
kubectl get pods -n mlops && curl -s localhost:5000/health   # pods Running, OK
```

Without `MLFLOW_TRACKING_URI`, MLflow silently writes to a local `mlflow.db` instead of the cluster.

---

## 6. Training and hyperparameter search (U2-R1)

Every training run logs params, `train_loss` per 10 steps and `epoch_loss` per epoch, **system metrics**
(CPU, memory, disk, network every 2 s via `psutil`), the train and test **datasets with digests**, the tags
`git_commit`, `random_seed`, `dvc_raw_md5`, `runner`, and the model with its environment (`requirements.txt`,
`python_env.yaml`) and signature.

```bash
dvc repro                               # prepare -> train -> evaluate, logged to the cluster MLflow
python scripts/hpo.py --trials 6        # Optuna (TPE, seeded): parent run hpo-optuna + 6 nested trial runs
```

The search scores trials on a validation split of the training data, never on the test set. Its best setting
(`max_length` 32, `batch_size` 16) is now the default; the current model uses `learning_rate` 1.2e-4.

| Run | Accuracy | Macro F1 | Notes |
| --- | --- | --- | --- |
| v1 (HPO best, lr 1.41e-4) | 0.887 | 0.855 | Registry v1 |
| v2 (lr 1.2e-4) | 0.880 | 0.846 | Registry v2, currently `@champion` |

In the UI: experiment `tweet-emotion` → the `hpo-optuna` run has an expand arrow with its 6 trials; select them →
**Compare** for the parameter chart. Each run's **System metrics** tab shows the resource charts.

---

## 7. Model Registry (U2-R2)

Registered model `tweet-emotion`, managed with aliases (stages are deprecated in MLflow 3):

| Alias | Meaning | Set by |
| --- | --- | --- |
| `@candidate` | Latest version that passed the quality gate | `python scripts/registry.py register` |
| `@champion` | The version the API serves: **the only serving source** | `python scripts/registry.py promote` |
| `@previous` | The champion before the last change | `promote` and `rollback` |

`register` reads `models/run_id.txt` and is idempotent (the same run never gets a second version).
In the UI: **Models → tweet-emotion** shows the versions, their aliases and the runs they came from.

---

## 8. Training Job in the cluster (U2-R3)

The training image runs as a Kubernetes Job. Hyperparameters come from the ConfigMap `train-params` (a full
`params.yaml`, mounted over `/app/params.yaml`); the MLflow address comes from Secret `mlflow-client`, which is
created by a command and never stored in Git.

```bash
# once: Secret with the in-cluster MLflow address
kubectl create secret generic mlflow-client -n mlops \
  --from-literal=MLFLOW_TRACKING_URI=http://mlflow.mlops.svc.cluster.local:5000

# training image for the current commit, pushed to GHCR and loaded into the node
TAG=sha-$(git rev-parse --short=7 HEAD)
docker build -f docker/Dockerfile --build-arg GIT_COMMIT=${TAG#sha-} -t ghcr.io/nsannn/mlops-nsannn/train:$TAG .
docker push ghcr.io/nsannn/mlops-nsannn/train:$TAG
kind load docker-image ghcr.io/nsannn/mlops-nsannn/train:$TAG --name mlops
sed -i "s|train:sha-[0-9a-f]*|train:$TAG|" k8s/train-job.yaml

# run (or re-run after changing a value in the ConfigMap part of the file)
kubectl delete job train -n mlops --ignore-not-found && kubectl apply -f k8s/train-job.yaml
kubectl logs -n mlops job/train -f
kubectl get job,pods -n mlops -l app=train             # Complete 1/1, pod Completed
```

The run appears in MLflow with the tag `runner=k8s-job` (measured: `Completed` in 104 s).

---

## 9. Inference API (U2-R4)

FastAPI service (`src/serve.py`, image `docker/Dockerfile.api`). At start-up it resolves
`models:/tweet-emotion@champion` to a version and downloads that model from MLflow; the image contains no model.
Deployment + Service in `k8s/base/api.yaml` (`ClusterIP`; reached through a port-forward), applied by Argo CD
(section 12).

```bash
kubectl port-forward -n mlops svc/emotion-api 8000:80        # keep running in its own terminal
curl -s localhost:8000/health                                # {"status":"ok",...,"version":"2",...}
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"text": "i am so happy today"}'
python scripts/load_test.py --n 100 --concurrency 1          # p50/p95/p99 and throughput
```

| What | Address |
| --- | --- |
| API | `http://localhost:8000` via port-forward; inside the cluster `http://emotion-api.mlops.svc` |
| Interactive docs | [http://localhost:8000/docs](http://localhost:8000/docs) |
| Load test results | [`docs/load-test.md`](docs/load-test.md): p95 7.2 ms (1 client), 100 req/s (4 clients) |

---

## 10. CI/CD, quality gate and rollback (U2-R5, U2-R6)

GitHub Actions ([`.github/workflows/ml-pipeline.yml`](.github/workflows/ml-pipeline.yml)) runs on GitHub's
runners and never connects to the cluster:

| Job | Pull request | Push to `main` |
| --- | --- | --- |
| `lint` → `test` | yes | `ruff check .`, `pytest` |
| `model-gate` | yes | data from the pinned HF revision (md5 checked against DVC) → `dvc repro` → gate |
| `build-push` | no | API image → Trivy (blocks CRITICAL) → GHCR `sha-<commit>` → bot commit of the tag to `k8s/base/api.yaml` |

The **quality gate** is set in `params.yaml` (`gate: metric: f1, threshold: 0.80`) and checked by
`scripts/check_model_gate.py` (exit 1 stops the pipeline). Changing the threshold retrains nothing.

**Deploy:** Argo CD applies the bumped manifest by itself (section 12). The registry half of a deployment, which Argo CD
can't see, stays a script:

```bash
git pull && ./scripts/deploy.sh   # gate -> register @candidate -> promote @champion -> wait for Argo CD sync -> restart -> smoke test
```

**Rollback** of the model (measured 17 s):

```bash
python scripts/registry.py rollback && kubectl rollout restart deployment/emotion-api -n mlops \
  && kubectl rollout status deployment/emotion-api -n mlops
```

Green and red runs, the rollback drill and the other rollback options: [`docs/cicd.md`](docs/cicd.md).
Actions: <https://github.com/nsannn/mlops-nsannn/actions>.

---

## 11. ONNX variant (U2-E4)

A second, slim API serves the same model exported to ONNX and quantized to INT8, next to the PyTorch API. It still
loads from the registry by alias (`models:/tweet-emotion-onnx@champion`); its image has no torch, no transformers
and no model inside.

```bash
# export the current tweet-emotion@champion, check parity / F1 against it, register tweet-emotion-onnx@champion
python scripts/export_onnx.py --no-register      # checks only
python scripts/export_onnx.py                    # checks + register (INT8; --variant fp32 for the unquantized one)

# slim image for the current commit, scanned, loaded into kind and deployed as emotion-api-onnx
TAG=sha-$(git rev-parse --short=7 HEAD)
docker build -f docker/Dockerfile.api-onnx -t ghcr.io/nsannn/mlops-nsannn/api-onnx:$TAG .
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy:0.76.1 image \
  --severity CRITICAL --ignore-unfixed --exit-code 1 ghcr.io/nsannn/mlops-nsannn/api-onnx:$TAG
kind load docker-image ghcr.io/nsannn/mlops-nsannn/api-onnx:$TAG --name mlops
sed -i "s|api-onnx:sha-[0-9a-f]*|api-onnx:$TAG|" k8s/base/api-onnx.yaml
# commit + push the new tag: Argo CD (section 12) rolls emotion-api-onnx out

kubectl port-forward -n mlops svc/emotion-api-onnx 8001:80      # own terminal
python scripts/load_test.py --url http://localhost:8001/predict --n 100 --concurrency 1
```

Result ([`docs/onnx-comparison.md`](docs/onnx-comparison.md)): image 415 MB instead of 1 940 MB, model 10.8 MiB
instead of 42.6 MiB, pod memory 140 MiB instead of 1 010 MiB, p95 4.0 ms instead of 6.6 ms (1 client) and
587 instead of 108 req/s (4 clients), macro F1 0.8461 instead of 0.8464.

---

## 12. Where to look

| What | Where |
| --- | --- |
| Runs, HPO tree, system metrics | MLflow [http://localhost:5000](http://localhost:5000) → Model training → `tweet-emotion` |
| Registered models and aliases | MLflow → **Models** → `tweet-emotion`, `tweet-emotion-onnx` |
| CI runs (green and red) | [GitHub Actions](https://github.com/nsannn/mlops-nsannn/actions) |
| Images | GHCR packages `mlops-nsannn/train`, `mlops-nsannn/api`; the ONNX image is built locally and loaded into kind |
| Measurements | [`docs/requirements.md`](docs/requirements.md), [`docs/load-test.md`](docs/load-test.md), [`docs/cicd.md`](docs/cicd.md), [`docs/onnx-comparison.md`](docs/onnx-comparison.md) |

---

## 13. Cleaning up

```bash
kind delete cluster --name mlops     # also deletes the PVC data (all MLflow runs and registered models)
```
