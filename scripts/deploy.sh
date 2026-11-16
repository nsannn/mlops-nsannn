#!/usr/bin/env bash
# scripts/deploy.sh - bring the cluster to the Git state (the step Argo CD would automate, U2-E5).
# Run after the CI run is green and `git pull` brought the bump commit:  ./scripts/deploy.sh
# gate -> register @candidate -> promote @champion -> kubectl apply -> rollout restart -> smoke test
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MLFLOW_TRACKING_URI:?export MLFLOW_TRACKING_URI=http://localhost:5000 first}"

test -z "$(git status --porcelain)" || { echo "working tree is dirty: commit first"; exit 1; }
git fetch -q origin main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" || { echo "not at origin/main: git pull first"; exit 1; }
dvc status -q || { echo "outputs differ from dvc.lock: run dvc repro or dvc pull"; exit 1; }

python scripts/check_model_gate.py        # the same gate as CI, on the model you are about to serve
python scripts/registry.py register       # models/run_id.txt -> registry version, alias @candidate
python scripts/registry.py promote        # @champion -> that version; the old champion keeps @previous

kubectl apply -f k8s/api.yaml             # the image tag from the CI bump commit
kubectl rollout restart deployment/emotion-api -n mlops   # pods read @champion only at start-up
kubectl rollout status deployment/emotion-api -n mlops --timeout=300s
kubectl run "smoke-$(date +%s)" -n mlops --rm -i --restart=Never --image=curlimages/curl:8.11.1 -- \
  curl -sf http://emotion-api/health
