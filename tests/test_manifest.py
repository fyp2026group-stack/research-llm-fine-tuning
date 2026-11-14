import json
from pathlib import Path

import pytest

from network_slm.manifest import ManifestError, build_manifest, sha256_file, validate_config


@pytest.fixture
def smoke_config():
    return {
        "schema_version": 1,
        "experiment_id": "stage_b_smoke",
        "run_kind": "smoke",
        "seed": 42,
        "model": {"id": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "unverified"},
        "datasets": {},
    }


def test_smoke_manifest_does_not_claim_execution(smoke_config, tmp_path):
    manifest = build_manifest(smoke_config, tmp_path)
    assert manifest["status"] == "registered_not_executed"
    assert manifest["git"]["revision"] is None
    assert manifest["configuration"]["seed"] == 42
    assert len(manifest["configuration_sha256"]) == 64
    assert manifest["dataset_files"] == {}


def test_deterministic_config_hash(smoke_config, tmp_path):
    a = build_manifest(smoke_config, tmp_path)
    b = build_manifest(json.loads(json.dumps(smoke_config)), tmp_path)
    assert a["configuration_sha256"] == b["configuration_sha256"]
    assert a["run_id"] != b["run_id"]


def test_research_refuses_unpinned_revision(smoke_config):
    smoke_config["run_kind"] = "research"
    smoke_config["datasets"] = {"train": "data/train.jsonl"}
    with pytest.raises(ManifestError, match="exact 40-character"):
        validate_config(smoke_config)


def test_missing_dataset_is_error(smoke_config, tmp_path):
    smoke_config["datasets"] = {"train": "data/nonexistent.jsonl"}
    with pytest.raises(ManifestError, match="Missing dataset"):
        build_manifest(smoke_config, tmp_path)


def test_rejects_escaping_dataset_path(smoke_config, tmp_path):
    smoke_config["datasets"] = {"train": "../../etc/passwd"}
    with pytest.raises(ManifestError, match="escapes repository"):
        build_manifest(smoke_config, tmp_path)


def test_hash_is_correct(tmp_path):
    p = Path(tmp_path / "example.txt")
    p.write_text("hello", encoding="utf-8")
    assert sha256_file(p) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
