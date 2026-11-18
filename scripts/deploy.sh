#!/usr/bin/env bash
# scripts/deploy.sh - the registry half of a deployment; Argo CD (U2-E5) applies the manifests from Git.
# Run after the CI run is green and `git pull` brought the bump commit:  ./scripts/deploy.sh
# gate -> register @candidate -> promote @champion -> wait for Argo CD sync -> rollout restart -> smoke test
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

# Argo CD applies Git: ask it to refresh now (instead of within ~3 min) and wait for the image named in Git.
kubectl annotate application emotion-api -n argocd argocd.argoproj.io/refresh=normal --overwrite
WANT=$(awk '/image: ghcr.io\/.*\/api:/ {print $2}' k8s/base/api.yaml)
for _ in $(seq 1 60); do
  [ "$(kubectl get deploy emotion-api -n mlops -o jsonpath='{.spec.template.spec.containers[0].image}')" = "$WANT" ] && break
  sleep 5
done
[ "$(kubectl get deploy emotion-api -n mlops -o jsonpath='{.spec.template.spec.containers[0].image}')" = "$WANT" ] \
  || { echo "Argo CD has not synced $WANT within 5 min: check 'kubectl get application emotion-api -n argocd'"; exit 1; }
kubectl rollout restart deployment/emotion-api -n mlops   # Argo CD can't see @champion move; pods read it at start-up
kubectl rollout status deployment/emotion-api -n mlops --timeout=300s
# Retries: right after a rollout the Service can still route to the old pod that is shutting down.
kubectl run "smoke-$(date +%s)" -n mlops --rm -i --restart=Never --image=curlimages/curl:8.11.1 -- \
  curl -sf --retry 5 --retry-delay 2 --retry-all-errors http://emotion-api/health
