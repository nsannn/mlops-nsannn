# Tweet Emotion Detection — Requirements v1.3

Date: 2026-10-21 | Author: Nojus Sandovas

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

## 3. Non-functional requirements

| ID     | Category    | Requirement                                   | KPI                              |
|--------|-------------|-----------------------------------------------|----------------------------------|
| NFR-01 | Quality     | Test accuracy                                 | ≥ 0.85                           |
| NFR-02 | Quality     | Macro F1 over 6 classes                       | ≥ 0.75                           |
| NFR-03 | Fairness    | Recall of the rarest class (surprise, ~4 %)   | ≥ 0.50                           |
| NFR-04 | Performance | Training run on CPU (4 threads, 8k tweets)    | ≤ 60 s                           |
| NFR-05 | Performance | Single-tweet inference latency (U2)           | p95 ≤ 50 ms on CPU               |
| NFR-06 | Size        | Saved model weights                           | ≤ 60 MB                          |
| NFR-07 | Security    | Secrets committed to Git                      | 0                                |

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


## 7. Change history

| Version | Date       | Author      | Changes |
|---------|------------|-------------|---------|
| 1.0     | 2026-10-21 |Nojus Sandovas | Initial document: problem, FR-01..04, NFR-01..07 |
| 1.1     | 2026-10-21 | Nojus Sandovas | MLflow + DVC pipeline, reproducibility NFR-08..10 |
| 1.2     | 2026-10-21 | Nojus Sandovas | Training image on GHCR: containerisation NFR-11..14 with measured values |
| 1.3     | 2026-10-21 | Nojus Sandovas | Created kind cluster with PostgreSQL and MLFlow NFR-15..17 |

