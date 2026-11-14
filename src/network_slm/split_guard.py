"""Prevent leakage of network episodes across experiment splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class SplitError(ValueError):
    """Dataset split integrity failure."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SplitError(f"{path}:{index} is not valid JSON") from error
            if not isinstance(record, dict):
                raise SplitError(f"{path}:{index} must be an object")
            records.append(record)
    return records


def validate_splits(splits: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    """Related reports must share one episode_id and remain in the same split."""
    seen: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split, records in splits.items():
        counts[split] = 0
        for index, record in enumerate(records, 1):
            episode = record.get("episode_id")
            if not isinstance(episode, str) or not episode.strip():
                raise SplitError(f"{split} row {index} needs nonempty episode_id")
            if episode in seen and seen[episode] != split:
                raise SplitError(f"Episode {episode!r} leaked between {seen[episode]} and {split}")
            seen[episode] = split
            counts[split] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Check episode-level split isolation")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    args = parser.parse_args()
    counts = validate_splits(
        {
            "train": load_jsonl(args.train),
            "validation": load_jsonl(args.validation),
            "test": load_jsonl(args.test),
        }
    )
    print("EPISODE_SPLIT_CHECK_PASS", counts)


if __name__ == "__main__":
    main()
