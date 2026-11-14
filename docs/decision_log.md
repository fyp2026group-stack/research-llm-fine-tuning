# Architectural decision records

## ADR-0001 — Auditable Python package (2026-11-14)

Keep reusable research logic in `src/network_slm/`, separate data/testbed/training/evaluation modules, and test the small utilities independently. A notebook-only project was rejected because hidden state makes experiments harder to reproduce.

## ADR-0002 — Existing GitHub repository and reviewed branches

Use [fyp2026group-stack/research-llm-fine-tuning](https://github.com/fyp2026group-stack/research-llm-fine-tuning). The repository was empty and public when inspected. Preserve its visibility and use feature branches and pull requests so `main` stays stable.

## ADR-0003 — Local provenance first

Use SHA-256 dataset file checksums and source-code commit IDs in run manifests, with optional MLflow for local tracking. Introduce DVC only if needed. This avoids unnecessary infrastructure while preserving traceability.

## ADR-0004 — Preserve the working CUDA environment

Do not reinstall Torch or CUDA in ordinary CPU CI. Model preflight scripts are optional and use the user's known-good `~/llm-research/.venv`.

## ADR-0005 — No unverified research claims

The selected research hypothesis has not yet been tested. Stage B includes infrastructure only, not a network benchmark or fine-tuned diagnostic model.
