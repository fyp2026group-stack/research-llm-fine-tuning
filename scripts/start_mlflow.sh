#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if ! command -v mlflow >/dev/null 2>&1; then
  echo 'MLflow is not installed; see docs/git_and_setup.md' >&2
  exit 1
fi
mkdir -p "$ROOT/artifacts/mlflow"
cd "$ROOT/artifacts/mlflow"
echo 'Starting local-only MLflow: http://127.0.0.1:5000'
exec mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
