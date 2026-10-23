# MLOps U1 — Tweet Emotion Detection

This project builds a reproducible training pipeline for a model that classifies tweets into 6 emotions
(sadness, joy, love, anger, fear, surprise). It uses Git, DVC, Docker and a local Kubernetes (`kind`) cluster,
with every experiment tracked in an MLflow server that runs inside the cluster.

- **Model:** [`google/bert_uncased_L-4_H-256_A-4`](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4) (11M parameters), fine-tuned on CPU
- **Data:** [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion), seeded 10k-tweet stratified sample
- **Submission:** tag `u1-final` · requirements in [`docs/requirements.md`](docs/requirements.md)

---

## 1. Running the project from zero

### Folder structure

The project follows the required structure: `src/`, `docker/`, `k8s/`, `docs/`.

| Path | Contents |
| --- | --- |
| `src/` | Pipeline code: `prepare_data.py`, `train.py`, `evaluate.py`, `model_utils.py` |
| `scripts/` | `fetch_data.py`: downloads the pinned dataset and takes the sample |
| `docker/` | `Dockerfile` (training image), `Dockerfile.mlflow` (MLflow server image) |
| `k8s/` | `kind-config.yaml` (cluster as code) and Kubernetes manifests |
| `docs/` | Requirements, image size, security scan, SBOM |
| `params.yaml`, `dvc.yaml`, `dvc.lock` | Hyperparameters, pipeline definition, pipeline lock (data and output hashes) |
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

Exp 3 is the default in `params.yaml`: it meets the ≤ 60 s training target (NFR-04) at a small accuracy cost.

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

## 5. Cleaning up

```bash
kind delete cluster --name mlops     # also deletes the PVC data (all MLflow runs)
```
