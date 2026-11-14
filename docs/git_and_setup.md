# Guided Ubuntu setup and GitHub workflow

This document teaches the reason for each step. **Do not delete or reinstall** the already validated CUDA installation at `~/llm-research/.venv`.

## 1. Clone the selected GitHub repository

```bash
cd ~/llm-research
git clone https://github.com/fyp2026group-stack/research-llm-fine-tuning.git
cd research-llm-fine-tuning
```

**Why:** `git clone` downloads the tracked source and its change history; there is no need to extract the previous Stage B ZIP. `main` is the stable branch. Work from a feature branch and use a pull request before merging.

If you are reviewing the Stage B changes before they are merged:

```bash
git fetch origin
git switch --track origin/feature/stage-b-research-foundation
```

## 2. Isolated CPU testing environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest -q
```

**Why:** An isolated environment avoids changing a working GPU setup. Editable installation exposes changes to source modules immediately. Ruff validates code style; pytest checks functions without needing NVIDIA hardware.

The development dependencies are pinned, but CUDA research dependencies will be recorded separately before actual training. A `pip freeze` file is an environment snapshot, **not** a portable dependency lock.

## 3. Register a test experiment manifest

```bash
network-slm-manifest --config configs/experiments/stage_b_preflight.json
```

The output is an ignored JSON file under `artifacts/runs`. A manifest records Git revision, configuration hash, library versions, and (for data-backed experiments) file checksums. The field `registered_not_executed` intentionally means **no model was run**.

## 4. Development workflow

```bash
git switch main
git pull --ff-only
git switch -c feature/mqtt-testbed-design
# Make approved changes, then inspect:
git status
git diff
git add <specific-paths>
git commit -m "docs: specify controlled MQTT testbed"
git push -u origin feature/mqtt-testbed-design
```

Open a GitHub pull request into `main`. Review the diff and CI result before merging. Do not commit .env files, tokens, datasets, model weights, or experiment outputs.

**Note:** The current GitHub repository is public; never upload private research data or unlicensed third-party content.

## 5. Optional check using the existing CUDA environment

```bash
deactivate
source ~/llm-research/.venv/bin/activate
python scripts/model_smoke.py
python scripts/capture_environment.py
```

The first script only runs an optional 4-bit model inference check and downloads the weights if missing. It does not train or evaluate the network-research idea. The results go to ignored `artifacts/`.

## 6. Optional local MLflow tracking

```bash
python3 -m venv ~/llm-research/.venv-mlflow
source ~/llm-research/.venv-mlflow/bin/activate
python -m pip install 'mlflow>=3.7,<4'
bash scripts/start_mlflow.sh
```

Then browse http://127.0.0.1:5000 on the same PC. MLflow is bound to loopback, not the wider network. Stage B adds the launcher; later training scripts can log metrics here.

## 7. Acceptance checks

Confirm 11 unit tests pass locally; CPU CI passes on the pull request; a provenance manifest can be generated; and no sensitive files were committed. Treat GPU training feasibility as **untested**, even though the earlier inference preflight passed.
