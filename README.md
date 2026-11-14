# Counterfactual Network SLM

**Stage B — reproducible research engineering foundation.** This repository is not yet a trained diagnostic model or a validated novelty claim.

## Research question

Can a small language model be trained to resist misleading network fault reports while remaining responsive to genuine changes in measured network telemetry?

We will evaluate evidence-sensitive fine-tuning against original prompting, conventional QLoRA, identical-data paired QLoRA, and simple diagnostic rules. The initial testbed will be controlled MQTT scenarios; no benchmark result is claimed at this stage.

## Getting started

On Ubuntu 22.04, clone this repository and use a **separate CPU development environment**, leaving the previously verified CUDA installation at `~/llm-research/.venv` unchanged.

```bash
git clone https://github.com/fyp2026group-stack/research-llm-fine-tuning.git
cd research-llm-fine-tuning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest -q
network-slm-manifest --config configs/experiments/stage_b_preflight.json
```

For a guided explanation of every command, read [the setup guide](docs/git_and_setup.md). See [research protocol](docs/research_protocol.md) and [decision log](docs/decision_log.md) for the scientific rationale.

## Current components

- `src/network_slm/manifest.py`: validates configurations and records code/data provenance without claiming experiments were executed.
- `src/network_slm/split_guard.py`: prevents one network episode from appearing across experiment splits.
- `tests/`: CPU-only unit tests.
- `scripts/model_smoke.py`: optional Qwen2.5-1.5B 4-bit GPU inference check; not a training or research result.
- `scripts/start_mlflow.sh`: optional loopback-only local experiment tracking.
- `.github/workflows/ci.yml`: lint, format, and tests on pushes and pull requests.

**Security:** never commit credentials, private network logs, raw datasets, downloaded models, or checkpoints. The GitHub repository was public when Stage B was initialized; review dataset and publication licenses before distributing research artifacts.

**Scientific status:** the novelty of the proposed training objective remains unverified. No MQTT fault injection, production diagnosis, or fine-tuning code has been implemented.
