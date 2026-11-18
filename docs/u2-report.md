# U2 Report — Tweet Emotion Detection

**Author:** Nojus Sandovas  
**Date:** 2026-11-18  
**Repo:** https://github.com/nsannn/mlops-nsannn · tag `u2-final` · requirements v1.6

---

## 1. What was done

One path from a commit to `/predict`, in the same kind cluster as U1. CI runs on GitHub's runners and never
connects to the cluster; the cluster follows Git:

```text
commit ──► GitHub Actions: lint → test → model-gate (data from pinned HF + md5 check, dvc repro, F1 ≥ 0.80)
                           → build-push (API image, Trivy, GHCR sha-<commit>, bot commit of the tag to k8s/api.yaml)
laptop ──► git pull && scripts/deploy.sh: gate → register @candidate → promote @champion → kubectl apply → restart
cluster ─► emotion-api pod: resolves models:/tweet-emotion@champion at start-up, loads that version from MLflow
```

* **U2-R1:** Every training run logs params, `train_loss` per 10 steps and `epoch_loss` per epoch, system metrics
  (psutil, every 2 s), the train and test datasets with digests, `git_commit` and `random_seed` tags, and the model
  with its environment and signature. `scripts/hpo.py` runs an Optuna (TPE, seeded) search as one parent run
  `hpo-optuna` with 6 nested trials, scored on a validation split; the best setting (`max_length` 32,
  `batch_size` 16, lr 1.41e-4) became the default.

  | Model | Change | Accuracy | Macro F1 | Registry |
  | --- | --- | --- | --- | --- |
  | v1 | HPO best, lr 1.41e-4 | 0.887 | 0.855 | `@previous` |
  | v2 | lr 1.2e-4 | 0.880 | 0.846 | `@champion` |

* **U2-R2:** `scripts/registry.py` registers gate-passing runs as versions of `tweet-emotion` and moves the aliases
  `@candidate`, `@champion` and `@previous`. The API reads `@champion` only; its image contains no model.
* **U2-R3:** `k8s/train-job.yaml` runs the training image as a Job: params from ConfigMap `train-params`, MLflow
  address from Secret `mlflow-client` (created by a command, not in Git). It finished `Completed` in 104 s; the run is
  tagged `runner=k8s-job` and has the same F1 as on the laptop.
* **U2-R4:** FastAPI (`src/serve.py`, `docker/Dockerfile.api`, 1 940 MB, non-root) with `/health` and `/predict`,
  Deployment + Service in `k8s/api.yaml`. Load test of 100 requests in the cluster (`docs/load-test.md`):

  | NFR | What | Target | Measured |
  | --- | --- | --- | --- |
  | NFR-05 | `/predict` p95, 1 client | ≤ 50 ms | **7.2 ms** |
  | NFR-21 | Throughput, 4 clients | ≥ 20 req/s | **100.3 req/s** |
  | NFR-06 | Model weights | ≤ 60 MB | 42.6 MiB |
  | NFR-22 | API ready after a rollout | ≤ 120 s | 10–11 s |

* **U2-R5:** `.github/workflows/ml-pipeline.yml`: `ruff`, 8 `pytest` unit tests (gate rules, API contract with a fake
  model), API image build, Trivy (blocks CRITICAL) and push to GHCR as `sha-<commit>`.
* **U2-R6:** The `model-gate` job reruns `dvc repro` and checks macro F1 ≥ 0.80 from `params.yaml`; only then is the
  image built and the new tag committed to `k8s/api.yaml`. 5 green runs took 7 min 56 s – 9 min 56 s (NFR-24).
  * **Red run #6** ([link](https://github.com/nsannn/mlops-nsannn/actions/runs/43957276366)): `learning_rate` 1e-6,
    1 epoch → `FAIL: f1=0.0842 < 0.8`, `build-push` skipped. No image tag, no bump commit, nothing registered,
    `/health` still v2, and `deploy.sh` refused. Reverted with `git revert` (run #7 green).
  * **Rollback drill:** `registry.py rollback` + `rollout restart`. Before: `champion: 2`, `/health` version 2.
    After: `champion: 1`, `/health` and `/predict` version 1, same image `api:sha-f325e5c`. The old model was
    answering **17 s** after the decision (NFR-23 ≤ 5 min). Details and the other rollback levels: `docs/cicd.md`.
* **U2-R7:** `docs/requirements.md` v1.4 → v1.6 (Git tags `v1.4`–`v1.6`): FR-05..06 and NFR-18..26, each with a
  measured value.

## 2. What did not work / technical issues and fixes

* **CI can't reach the cluster, MLflow or `~/dvcstore`:** CI rebuilds the data
  from the pinned Hugging Face revision (md5 equal to the DVC pointer) and logs to a throwaway SQLite MLflow; the
  served model is trained against the cluster MLflow from the same commit and gated again by `deploy.sh`.
* **CI F1 varies by runner** (0.8451–0.8570): GitHub's machines have different CPUs, so float results differ
  slightly; deterministic settings only guarantee equality on the same hardware.
* **GitHub account locked for billing:** the first CI run never started; it ran after the lock was lifted.
* **Trivy blocked the API image:** mlflow 3.15.2 has CVE-2026-116692 (CRITICAL). The API image uses the 3.17.0
  client, which works with the 3.15.2 server. The server and training images still have 3.15.2: accepted risk.
* **CPU throttling:** p95 was 92 ms because torch started 10 threads in a 2-CPU pod. Fixed with
  `OMP_NUM_THREADS=2` and one inference at a time (p95 7.2 ms).
* **Targets vs. reality:** NFR-04 was relaxed from 60 s to 120 s, because the tuned params and fuller tracking made
  training 88 s. A first second-model attempt (lr 2e-4) passed the gate by only 0.007, so lr 1.2e-4 was used.
* **Smaller issues:** the API failed to start without `MLFLOW_TRACKING_URI` (MLflow used an empty local DB); the
  smoke test ran before the old pod was gone (added retries); a Windows-side edit dropped the executable bit of
  `deploy.sh`; ruff 0.17 enables import-sort and blind-except rules by default. The HPO-tuned run was trained before
  its params were committed, so its `git_commit` tag points one commit earlier; later params are committed first.

## 3. Time spent

* Tracking, HPO and registry: **~3 h**
* API, image and training Job: **~2.5 h**
* CI/CD, gate, deploy, rollback: **~4.5 h**
* Load test, documentation and report: **~2 h**
* **Total:** ~12 h
