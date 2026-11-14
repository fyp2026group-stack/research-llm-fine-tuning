import pytest

from network_slm.split_guard import SplitError, load_jsonl, validate_splits


def test_valid_variants_remain_in_one_split():
    counts = validate_splits(
        {
            "train": [{"episode_id": "e1"}, {"episode_id": "e1"}],
            "validation": [{"episode_id": "e2"}],
            "test": [{"episode_id": "e3"}],
        }
    )
    assert counts == {"train": 2, "validation": 1, "test": 1}


def test_reject_leakage_of_same_episode():
    with pytest.raises(SplitError, match="leaked"):
        validate_splits({"train": [{"episode_id": "e1"}], "test": [{"episode_id": "e1"}]})


def test_missing_episode_id_is_error():
    with pytest.raises(SplitError, match="episode_id"):
        validate_splits({"train": [{"ticket": "broker failed"}]})


def test_read_jsonl(tmp_path):
    p = tmp_path / "sample.jsonl"
    p.write_text('{"episode_id":"a"}\n\n{"episode_id":"b"}\n', encoding="utf-8")
    assert [x["episode_id"] for x in load_jsonl(p)] == ["a", "b"]


def test_bad_jsonl_rejected(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"episode_id":\n', encoding="utf-8")
    with pytest.raises(SplitError, match="valid JSON"):
        load_jsonl(p)
