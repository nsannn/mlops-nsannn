# U1 Report — Tweet Emotion Detection

**Author:** Nojus Sandovas  
**Date:** 2026-10-23  
**Repo:** https://github.com/nsannn/mlops-nsannn · tag `u1-final`

---

## 1. What was done

* **U1-R1:** Repository with the required structure (`src/`, `docker/`, `k8s/`, `docs/`), a README that runs the
  project from zero (ports, logins, where to look), and a versioned `docs/requirements.md` (v1.0 → v1.3, Git tags
  `v1.0`–`v1.3`): 4 FR and 17 NFR with KPIs, including reproducibility NFR-08..10.
* **U1-R2:** The model is `google/bert_uncased_L-4_H-256_A-4` (11M parameters), fine-tuned on CPU to classify tweets
  into 6 emotions. The data is a seeded 10k-tweet stratified sample of `dair-ai/emotion`, both pinned by revision.
  A 3-stage DVC pipeline (`prepare` → `train` → `evaluate`) runs from `params.yaml`, with `dvc.lock` committed and a
  local DVC remote (`~/dvcstore`). There are 3 MLflow runs with different hyperparameters, each tagged with
  `git_commit` and `dvc_raw_md5`. `reproduce.py` re-created a run in ~84 s with an identical result (Δ = 0).

  | Run | Change | Accuracy | Macro F1 | Run time |
  | --- | --- | --- | --- | --- |
  | exp 1 | lr 1e-4, max_length 32 | 0.874 | 0.840 | 1.1 min |
  | exp 2 | lr 3e-4 | 0.858 | 0.821 | 1.0 min |
  | exp 3 (default) | max_length 24 | 0.860 | 0.834 | 52 s |

* **U1-R3:** Multi-stage training image (`docker/Dockerfile`), non-root (`uid=1000`), with pinned dependencies and
  model weights baked in, so it trains offline. It is published as `ghcr.io/nsannn/mlops-nsannn/train:sha-af2c8df`
  (public, linked to the repo). Size 2 070 MB; Trivy found 0 CRITICAL and 46 HIGH, all reviewed in
  `docs/security-scan.md`, plus an SPDX SBOM. Inside the container training gave the same 0.874 / 0.840 as on the laptop.
* **U1-R4:** A `kind` cluster from `k8s/kind-config.yaml`, with namespace `mlops` containing PostgreSQL (metadata)
  and an MLflow 3.15.2 Tracking Server (Deployment + NodePort Service), each with its own PVC. MLflow is reachable at
  `http://localhost:5000`. All runs are logged to the cluster (`mlflow-artifacts:/` URIs); they survived deleting
  both pods and a laptop reboot. MLflow was ready 61 s after `kubectl apply`.

## 2. What did not work / technical issues and fixes

* **`log_model` crashed with `No module named 'torchvision'`:** with CPU-only torch, MLflow's automatic dependency
  detection failed. Fixed by passing `pip_requirements=requirements-train.txt` explicitly.
* **The MLflow pod never started:** three separate causes.
  * The init container hung because `pg_isready` had no user for UID 1000. Fixed with `-U mlflow`.
  * MLflow crashed with `No module named 'psycopg'` because SQLAlchemy 2.1 maps `postgresql://` to psycopg v3.
    Fixed with a `postgresql+psycopg2://` URI.
* **Targets vs. reality:**
  * Training at `max_length` 32 took ~66 s, over the 60 s target (NFR-04), so `max_length` 24 (52 s, −1.4 pp
    accuracy) became the default.
  * The image size target was set from the measured baseline (≤ 2 200 MB), because CPU-only torch dominates.
  * The `cryptography` HIGH CVE cannot be fixed (`mlflow==3.15.2` requires `<50`); it is documented as an accepted risk.

## 3. Time spent

* Environment, repo and requirements (v1.0–v1.1): **~1 h**
* Data, DVC pipeline, training code and reproduction: **~2 h**
* Docker image, Trivy scan, GHCR: **~1.5 h**
* Kubernetes manifests, PostgreSQL + MLflow in the cluster, debugging: **~2.5 h**
* Experiments, evidence, README and report: **~1.5 h**
* **Total:** ~8.5 h
