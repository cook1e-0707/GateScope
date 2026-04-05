from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def artifact_root(script_file: str | Path) -> Path:
    return Path(script_file).resolve().parents[1]


def print_stage_banner(title: str, status: str, summary: str) -> None:
    line = "=" * 80
    print(line)
    print(title)
    print(line)
    print(f"Status : {status}")
    print(f"Summary: {summary}")
    print()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"JSONL line {line_number} must decode to an object.")
            rows.append(parsed)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def list_json_files(path: str | Path) -> list[Path]:
    input_path = Path(path)
    if input_path.is_file():
        return [input_path]
    return sorted(child for child in input_path.glob("*.json") if child.is_file())
