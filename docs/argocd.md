# Argo CD: GitOps deployment of the inference APIs (U2-E5)

Date: 2026-11-18 (times UTC) · Argo CD v3.5.3 · Application [`deploy/argocd/application.yaml`](../deploy/argocd/application.yaml)

## Setup

Argo CD runs inside the kind cluster (namespace `argocd`, 7 pods, about 0.8 GiB) and pulls from GitHub, an
outbound connection, so CI still never connects to the cluster. It manages only the two inference APIs:

| Path | Contents |
| --- | --- |
| `k8s/base/` | `api.yaml` (PyTorch API), `api-onnx.yaml` (ONNX API), `kustomization.yaml` |
| `deploy/overlays/production/` | Kustomize overlay: namespace `mlops`, resources `k8s/base` |
| `deploy/argocd/application.yaml` | Application `emotion-api`: repo `main`, path = the overlay, `automated: {prune, selfHeal}` |

MLflow, PostgreSQL, the training Job and `kind-config.yaml` stay outside it (pointing it at `k8s/` would make it
manage and prune them). The registry alias is outside its view too: `@champion` moving is not a Git change, so
`scripts/deploy.sh` still promotes and restarts the pods; it no longer runs `kubectl apply` but waits for the sync.

```bash
kubectl apply -n argocd --server-side --force-conflicts \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/v3.5.3/manifests/install.yaml
kubectl apply -f deploy/argocd/application.yaml
```

First sync (commit `17e1816`): `Synced/Healthy` after 79 s; all six objects (2 ConfigMaps, 2 Services,
2 Deployments) Synced. It rolled `emotion-api` to the newer image named in Git (`api:sha-a296936`) and left the
ONNX API, MLflow and PostgreSQL untouched.

## Drill 1: auto-sync, no manual step

| Time (UTC) | Event |
| --- | --- |
| 22:21:43 | Push of `17e1816` (layout change) starts CI run #10 |
| 22:30:28 | Run #10 green; bot commit `0a592da` "ci: bump api to sha-17e1816" |
| 22:36:37 | Argo CD detects the new revision and syncs (operation 1 s) |
| 22:37:43 | New pod Ready |

**374 s** from the bump commit to the new image applied, **435 s** to the pod Ready, with nothing run by hand.
The delay is Argo CD's polling (every 2 min + up to 1 min jitter) plus the repo server's cached view of the branch.
`deploy.sh` therefore requests a hard refresh (`argocd.argoproj.io/refresh=hard`), which syncs immediately.

## Drill 2: self-heal

`kubectl scale deployment/emotion-api-onnx -n mlops --replicas=2` at 22:27:20: Argo CD set it back to the Git
value (1 replica) **within 1 s**; the application stayed `Synced/Healthy`. Manual changes can't make the cluster
drift from Git.

## Drill 3: rollback through Argo CD, then Git

| Step | Command (`argocd --core`, via kubeconfig) | Result |
| --- | --- | --- |
| Before | `argocd app history emotion-api` | id 0 = `17e1816` (image `sha-a296936`), id 1 = `0a592da` (`sha-17e1816`, running) |
| 1 | `argocd app set emotion-api --sync-policy none` | auto-sync off (Argo CD refuses a rollback while it is on) |
| 2 | `argocd app rollback emotion-api 0` (22:43:19) | image `sha-a296936`, rollout done after 12 s, `/health` answering after **17 s**; app `OutOfSync/Healthy` (Git still differs) |
| 3 | `git revert --no-edit 0a592da && git push` → `4648bbe` | Git names `sha-a296936` again; `paths-ignore` keeps CI from bumping it back |
| 4 | hard refresh | `Synced/Healthy` at `4648bbe` |
| 5 | `argocd app set emotion-api --sync-policy automated --self-heal --auto-prune` | back to normal |
