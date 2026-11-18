# CI/CD: quality gate, deployment and rollback
## Pipeline


Never connects to the cluster (GitHub can't reach the local kind cluster or its MLflow). Since U2-E5 the cluster
is brought to the Git state by Argo CD, which runs inside it and pulls from GitHub (`docs/argocd.md`); before that,
`scripts/deploy.sh` ran `kubectl apply` (runs #2–#7 below).

| Job | Runs on | Pull request | Push to `main` |
| --- | --- | --- | --- |
| `lint` → `test` | GitHub | yes | `ruff check .`, `pytest` (8 tests, fake model) |
| `model-gate` | GitHub | yes | `fetch_data.py` (pinned HF revision) → md5 check against `dataset.parquet.dvc` → `dvc repro` → `check_model_gate.py` (macro F1 ≥ 0.80 from `params.yaml`) |
| `build-push` | GitHub | no | API image → Trivy (fail on CRITICAL) → GHCR `sha-<commit>` → bot commit of the tag to `k8s/base/api.yaml` (`k8s/api.yaml` before U2-E5) |
| Argo CD | cluster | – | polls `main` (~3 min), applies `deploy/overlays/production` (both APIs), self-heals drift |
| `deploy.sh` | laptop, after `git pull` | – | gate on local `metrics.json` → `registry.py register` (@candidate) → `promote` (@champion, old → @previous) → wait for the Argo CD sync → `rollout restart` → smoke test |

Each job `needs` the previous one, so a red gate skips everything after it: no image, no bump commit,
nothing to deploy. CI's MLflow is a throwaway SQLite file; the served model is trained against the cluster
MLflow from the same commit and gated again by `deploy.sh`.

## Run log (newest first, times UTC, 2026-11-16)

| Run | Commit | Trigger | Result | Gate (CI) | Duration | Serving afterwards |
| --- | --- | --- | --- | --- | --- | --- |
| – | – | rollback drill (`registry.py rollback` + restart) | v2 → v1 in **17 s** | – | – | v1, then swapped back to v2 |
| [#7](https://github.com/nsannn/mlops-nsannn/actions/runs/43958191466) | `584e132` | revert of the bad model | green | PASS 0.8507 | 9 min 56 s | v2 (unchanged) |
| [#6](https://github.com/nsannn/mlops-nsannn/actions/runs/43957276366) | `87f46e6` | `learning_rate: 1e-6`, `epochs: 1` | **red**: `model-gate` failed, `build-push` skipped | **FAIL 0.0842** | 3 min 57 s | v2 (unchanged) |
| [#5](https://github.com/nsannn/mlops-nsannn/actions/runs/43933911770) | `f325e5c` | `learning_rate: 1.2e-4` (new model) | green + `deploy.sh` | PASS 0.8570 | 9 min 24 s | **v2** (local F1 0.8464) |
| [#4](https://github.com/nsannn/mlops-nsannn/actions/runs/43928984144) | `6ec914b` | executable bit on `deploy.sh` | green | PASS 0.8451 | 8 min 23 s | v1 |
| [#3](https://github.com/nsannn/mlops-nsannn/actions/runs/43926441197) | `3ce8977` | smoke-test retry | green + `deploy.sh` | PASS 0.8517 | 7 min 56 s | v1 |
| [#2](https://github.com/nsannn/mlops-nsannn/actions/runs/43920035899) | `ce4bcb9` | full pipeline added | green + `deploy.sh` | PASS 0.8517 | 8 min 37 s | v1 (local F1 0.8552) |
| [#1](https://github.com/nsannn/mlops-nsannn/actions/runs/43908797372) | `6813243` | CI only (lint, test, build-push) | green after a re-run | – | – | – |

## Red run: the gate blocks a bad model (#6)

`params.yaml` was pushed with `learning_rate: 0.000001`, `epochs: 1`, without training locally. CI trained it:
accuracy 0.338, macro F1 0.0842, and the gate printed `FAIL: f1=0.0842 < 0.8`. Checked afterwards:

| Check | Result |
| --- | --- |
| Bump commit for `87f46e6` | none: the last change to `k8s/api.yaml` is `9c1ccbb` (`sha-f325e5c`) |
| GHCR `api` tags | no `sha-87f46e6` |
| Registry | `champion: 2`, `previous: 1`: nothing registered |
| Running API | `/health` → `"version":"2"`, image `api:sha-f325e5c` |
| `./scripts/deploy.sh` | refused: `outputs differ from dvc.lock`, exit 1 |

Undone with `git revert` (`584e132`), which made `main` green again (#7) and changed nothing in the cluster.


## Observations

- **CI's F1 varies by runner.** Same commit, data md5, params and seed gave 0.8517, 0.8517 and 0.8451 for v1
  (laptop: 0.8552 every time). GitHub schedules jobs on machines with different CPUs, and floating-point
  results differ slightly between CPU models; deterministic settings only make runs identical on the same
  hardware. All values are well above the gate.
- **Pipeline duration:** 7 min 56 s – 9 min 56 s from push to the bump commit (installing torch and training
  dominate); `deploy.sh` adds 1.5–2.7 min.
