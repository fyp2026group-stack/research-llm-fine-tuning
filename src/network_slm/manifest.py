"""Generate research provenance manifests without executing model experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ManifestError(ValueError):
    """Invalid configuration or incomplete research provenance."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_info(root: Path) -> dict[str, Any]:
    revision = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain", "--untracked-files=normal")
    return {
        "revision": revision,
        "dirty": bool(status) if status is not None else None,
    }


def validate_config(cfg: dict[str, Any]) -> None:
    if not isinstance(cfg, dict):
        raise ManifestError("Config must be a JSON object")
    if cfg.get("schema_version") != 1:
        raise ManifestError("schema_version must be 1")
    name = cfg.get("experiment_id")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{2,63}", name):
        raise ManifestError("experiment_id must be a lowercase slug, 3-64 characters")
    if cfg.get("run_kind") not in {"smoke", "research"}:
        raise ManifestError("run_kind must be smoke or research")
    model = cfg.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("id"), str):
        raise ManifestError("model.id must be a string")
    if not model["id"].strip():
        raise ManifestError("model.id cannot be empty")
    if cfg["run_kind"] == "research":
        revision = model.get("revision")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ManifestError("Research run requires exact 40-character model commit hash")
        datasets = cfg.get("datasets")
        if not isinstance(datasets, dict) or not datasets:
            raise ManifestError("Research run requires dataset file paths")
    datasets = cfg.get("datasets", {})
    if not isinstance(datasets, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in datasets.items()
    ):
        raise ManifestError("datasets must map split names to nonempty file paths")
    if "seed" in cfg and (isinstance(cfg["seed"], bool) or not isinstance(cfg["seed"], int)):
        raise ManifestError("seed must be an integer")


def build_manifest(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    validate_config(cfg)
    root = root.resolve()
    git = git_info(root)
    if cfg["run_kind"] == "research" and (not git["revision"] or git["dirty"] is not False):
        raise ManifestError("Research run requires a committed, clean Git repository")

    hashes = {}
    for split, raw_path in cfg.get("datasets", {}).items():
        p = Path(raw_path)
        if p.is_absolute():
            raise ManifestError(f"Dataset path for {split} must be repository-relative")
        path = (root / p).resolve()
        if not path.is_relative_to(root):
            raise ManifestError(f"Dataset path for {split} escapes repository")
        if not path.is_file():
            raise ManifestError(f"Missing dataset: {p}")
        hashes[split] = {"path": p.as_posix(), "sha256": sha256_file(path)}

    libs = ["torch", "transformers", "accelerate", "peft", "trl", "bitsandbytes", "mlflow"]
    installed = {}
    for name in libs:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
    now = datetime.now(timezone.utc)
    run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    return {
        "manifest_schema_version": 1,
        "run_id": run_id,
        "created_utc": now.isoformat(),
        "run_kind": cfg["run_kind"],
        "experiment_id": cfg["experiment_id"],
        "git": git,
        "configuration": cfg,
        "configuration_sha256": hashlib.sha256(
            json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "dataset_files": hashes,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "libraries": installed,
        },
        "status": "registered_not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Register experiment provenance (no training)")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("artifacts/runs"))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = build_manifest(cfg, args.root)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    destination = out / f"{manifest['run_id']}.json"
    with destination.open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Manifest created: {destination}")
    print("Status: registered_not_executed (this does not run a model)")


if __name__ == "__main__":
    main()
