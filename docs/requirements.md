# Tweet Emotion Detection — Requirements v1.7

Date: 2026-11-18 | Author: Nojus Sandovas

## 1.1 Problem description
Create a API which can help social-media support team identify tweets and the emotions they encourage. 6 emotions will be tracked and identified (sadness, joy, love, anger, fear, surprise), build MLOps platform around it. Target: accuracy ≥ 0.85 and macro F1 ≥ 0.75 on held-out tweets,
with every experiment reproducible from Git + DVC + MLflow.


## 1.2 Data and model

- Data: [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) @ `cab853a`, seeded 10k stratified
  sample; licensed for educational and research use only.
- Model: [`google/bert_uncased_L-4_H-256_A-4`](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4) @ `387825c`
  (11M parameters, Apache-2.0), fine-tuned on CPU.

## 2. Functional requirements

| ID    | Requirement                                                                 | Priority |
|-------|-----------------------------------------------------------------------------|----------|
| FR-01 | Classify a tweet into one of 6 emotions and return the label name and probability | Must |
| FR-02 | `dvc repro` builds the model from pinned raw data: prepare → train → evaluate | Must |
| FR-03 | Every training run logs params, metrics, dataset, `git_commit` and `dvc_raw_md5` tags and the model to the cluster MLflow | Must |
| FR-04 | Training runs from a container image (and as a Kubernetes Job) | Should |
| FR-05 | Every model that passes the quality gate is registered as a version of `tweet-emotion`; serving reads `models:/tweet-emotion@champion` only (v1.4) | Must |
| FR-06 | `POST /predict` returns the emotion label, its probability, all 6 scores and the serving model version; `GET /health` reports the loaded version (v1.5) | Must |

## 3. Non-functional requirements

| ID     | Category    | Requirement                                   | KPI                              |
|--------|-------------|-----------------------------------------------|----------------------------------|
| NFR-01 | Quality     | Test accuracy                                 | ≥ 0.85                           |
| NFR-02 | Quality     | Macro F1 over 6 classes                       | ≥ 0.75                           |
| NFR-03 | Fairness    | Recall of the rarest class (surprise, ~4 %)   | ≥ 0.50                           |
| NFR-04 | Performance | Training run on CPU (4 threads, 8k tweets)    | ≤ 120 s (≤ 60 s until v1.3; see note) |
| NFR-05 | Performance | Single-tweet inference latency (U2)           | p95 ≤ 50 ms on CPU               |
| NFR-06 | Size        | Saved model weights                           | ≤ 60 MB                          |
| NFR-07 | Security    | Secrets committed to Git                      | 0                                |

**NFR-04 change (v1.4):** the U2 changes to the training process made the 60 s target no longer sustainable.
The Optuna search (6 nested trials) found better parameters (`max_length` 32, `batch_size` 16, learning rate 1.41e-4)
that raise macro F1 from 0.834 to 0.855 and accuracy from 0.860 to 0.887, but longer sequences and smaller batches mean
more computation per epoch. Each run also does more tracking (system metrics, two dataset inputs, a model signature).
A full training run went from 52 s to 88 s, so the
target is set to ≤ 120 s, which still fits inside the reproduction budget of NFR-08 (≤ 2 min).

## 3.1 Experiment tracking and reproducibility NFR (v1.1)

| ID     | Category        | Requirement                                                              | KPI |
|--------|-----------------|--------------------------------------------------------------------------|-----|
| NFR-08 | Reproducibility | Any historical run is re-created by `python reproduce.py <run_id>`       | ≤ 2 min (no data download), Δaccuracy ≤ 0.005 |
| NFR-09 | Traceability    | Every run has an MLflow run ID plus `git_commit` and data-md5 tags; every logged model has a `model_id` | 100 % of runs |
| NFR-10 | Data version    | Training data version pinned by DVC (`dvc.lock` md5 = the run's `dvc_raw_md5`) | 100 % of runs |

### 3.2 Containerisation (v1.2)
| ID | Category | Requirement | KPI | Measured | Evidence |
| --- | --- | --- | --- | --- | --- |
| NFR-11 | Size | Training image size (CPU-only torch dominates) | ≤ 2 200 MB | 2 070 MB | `image-size.md` |
| NFR-12 | Security | CRITICAL CVEs in the published image | 0 | 0 (46 HIGH reviewed) | `security-scan.md`
| NFR-13 | Security | Container runtime user | UID ≠ 0 | UID 1000 | `docker run --rm ghcr.io/nsannn/mlops-nsannn/train:sha-af2c8df id -u` |
| NFR-14 | Traceability | Published image tags | 100 % `sha-<commit>`, never `latest` | `sha-af2c8df` | GHCR page |


### 3.3 Kubernetes and MLflow server (v1.3)
| ID | Category | Requirement | KPI | Measured | Evidence |
| --- | --- | --- | --- | --- | --- |
| NFR-15 | Durability | Runs and artifacts after deleting the MLflow and PostgreSQL pods | 100 % kept | 100 % | `kubectl delete pod` test |
| NFR-16 | Availability | MLflow ready after `kubectl apply` (images present) | ≤ 120 s | 61 s | `kubectl wait` |
| NFR-17 | Tracking | Laptop runs stored on the cluster server | 100 %, no local `mlruns/` | `mlflow-artifacts:/` URIs | MLflow UI |

### 3.4 Training tracking and registry (v1.4)
| ID | Category | Requirement | KPI | Measured | Evidence |
| --- | --- | --- | --- | --- | --- |
| NFR-18 | Tuning | Hyperparameter search as nested runs | ≥ 5 trials in one parent run | 6 trials (best val F1 0.805) | MLflow run `hpo-optuna` |
| NFR-19 | Traceability | Every pipeline run has params, step metrics, system metrics, dataset digest, seed and commit tags, model environment | 100 % of runs | 100 % | Run page (e.g. `52fdd3aa`) |

### 3.5 Training Job and inference API (v1.5)
| ID | Category | Requirement | KPI | Measured | Evidence |
| --- | --- | --- | --- | --- | --- |
| NFR-20 | Operability | Training Job in the cluster, params from ConfigMap, MLflow from Secret | `Completed`, ≤ 5 min | `Completed`, 104 s (F1 0.855, same as the laptop run) | `kubectl get job train -n mlops`, run `e96a23db` (`runner=k8s-job`) |
| NFR-21 | Performance | API throughput, 100 requests, 4 concurrent clients | ≥ 20 req/s | 100.3 req/s (p95 71 ms) | `docs/load-test.md` |
| NFR-22 | Availability | API ready after a rollout, model download included | ≤ 120 s | 10–11 s | `kubectl rollout status` |

### 3.6 Inference and delivery (v1.6)
| ID | Category | Requirement | KPI | Measured | Evidence |
| --- | --- | --- | --- | --- | --- |
| NFR-05 | Performance | `/predict` latency, 100 requests, 1 client, in the cluster | p95 ≤ 50 ms | p95 7.2 ms (p50 6.0 ms) | `docs/load-test.md` |
| NFR-06 | Size | Saved model weights | ≤ 60 MB | 42.6 MiB (44.7 MB) | `docs/load-test.md` |
| NFR-23 | Recoverability | Model rollback until `/health` reports the previous version | ≤ 5 min | 17 s (v2 → v1) | `docs/cicd.md`, rollback drill |
| NFR-24 | Delivery | CI/CD pipeline duration, push to the image-tag commit | ≤ 15 min | 7 min 56 s – 9 min 56 s (5 green runs) | `docs/cicd.md`, GitHub Actions |
| NFR-25 | Quality | A model below the gate (macro F1 ≥ 0.80) is never built, registered or deployed | 100 % blocked | Blocked: run #6, F1 0.0842, `build-push` skipped, nothing deployed | `docs/cicd.md`, red run |
| NFR-26 | Security | Fixable CRITICAL CVEs in the API image | 0 (Trivy gate in CI) | 0 | `build-push` job, Trivy step |

Throughput (NFR-21) is measured in 3.5: 100.3 req/s with 4 clients. NFR-05 and NFR-21 were met only after fixing CPU
throttling in the API pod (`OMP_NUM_THREADS=2`, one inference at a time); before the fix p95 was 92 ms
(`docs/load-test.md`).

**Known risk:** mlflow 3.15.2 has CVE-2026-116692 (CRITICAL, fixed in 3.17.0), published after the U1 scan. The API
image uses mlflow 3.17.0 (NFR-26), but the MLflow server image and the training image still contain 3.15.2, so NFR-12
no longer holds for the training image.

### 3.7 ONNX inference variant (v1.7, U2-E4)
| ID | Category | Requirement | KPI | Measured | Evidence |
| --- | --- | --- | --- | --- | --- |
| NFR-27 | Size | ONNX inference image (no torch, no transformers) | ≤ 600 MB | 415 MB (PyTorch API: 1 940 MB) | `docs/onnx-comparison.md` |
| NFR-28 | Performance | ONNX `/predict` p95, 100 requests, 1 client, in the cluster | ≤ PyTorch p95 | 4.0 ms vs 6.6 ms | `docs/onnx-comparison.md` |
| NFR-29 | Quality | ONNX (INT8) model vs the PyTorch source: probability parity and macro F1 | parity ≤ 1e-4, ΔF1 ≤ 0.005 | 7.2e-7, ΔF1 0.0003 | `scripts/export_onnx.py`, run `04bf43a1` |


## 7. Change history

| Version | Date       | Author      | Changes |
|---------|------------|-------------|---------|
| 1.0     | 2026-10-21 |Nojus Sandovas | Initial document: problem, FR-01..04, NFR-01..07 |
| 1.1     | 2026-10-21 | Nojus Sandovas | MLflow + DVC pipeline, reproducibility NFR-08..10 |
| 1.2     | 2026-10-21 | Nojus Sandovas | Training image on GHCR: containerisation NFR-11..14 with measured values |
| 1.3     | 2026-10-21 | Nojus Sandovas | Created kind cluster with PostgreSQL and MLFlow NFR-15..17 |
| 1.4     | 2026-11-10 | Nojus Sandovas | Full run tracking, Optuna HPO, Model Registry with @champion: FR-05, NFR-18..19. NFR-04 target 60 s → 120 s: the tuned training process (HPO params, fuller tracking) no longer fits 60 s; measured 88 s |
| 1.5     | 2026-11-16 | Nojus Sandovas | Training as a Kubernetes Job (ConfigMap params, Secret MLflow address) and the inference API: FR-06, NFR-20..22 (NFR-20 measured) |
| 1.6     | 2026-11-16 | Nojus Sandovas | CI/CD with quality gate and rollback, measured inference NFRs: NFR-05, NFR-06, NFR-21, NFR-22 measured; new NFR-23..26; mlflow CVE noted as a known risk |
| 1.7     | 2026-11-18 | Nojus Sandovas | U2-E4 ONNX variant (INT8, slim image without torch, `tweet-emotion-onnx@champion`): NFR-27..29 measured |

