#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from artifact_utils import print_stage_banner
from collect.common import collect_responses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect official baseline responses from an OpenAI-compatible endpoint."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ARTIFACT_ROOT / "configs/config.example.yaml",
        help="YAML config with provider definitions.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        required=True,
        help="JSONL probe file, for example prompts/AIME_2024.jsonl.",
    )
    parser.add_argument("--provider-key", default="baseline", help="Provider key in the config file.")
    parser.add_argument("--model", required=True, help="Model name passed to the endpoint.")
    parser.add_argument("--repetitions", type=int, default=None, help="Number of calls per probe. Defaults to the paper protocol in the config.")
    parser.add_argument("--max-prompts", type=int, default=None, help="Optional prompt cap for smoke tests.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature. Defaults to the paper protocol in the config.")
    parser.add_argument("--timeout-seconds", type=float, default=None, help="Total per-request timeout across retries.")
    parser.add_argument("--spacing-hours", type=float, default=None, help="Minimum time gap between repetitions of the same probe.")
    parser.add_argument("--max-retries", type=int, default=None, help="Maximum retry count after the first attempt.")
    parser.add_argument("--attempt-timeout-seconds", type=float, default=None, help="Optional timeout cap for each individual HTTP attempt.")
    parser.add_argument("--max-output-tokens", type=int, default=None, help="Optional max token budget for compatible models.")
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=ARTIFACT_ROOT / "outputs/raw/official_records.jsonl",
        help="Normalized JSONL output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print_stage_banner(
        "Collect official baseline responses",
        "paper-aligned implementation",
        "Send the DCC prompt contract to the official API using the paper sampling protocol and record normalized JSONL logs.",
    )

    total_records, success_count = collect_responses(
        prompt_file=args.prompt_file.resolve(),
        output_jsonl=args.output_jsonl.resolve(),
        config_path=args.config.resolve(),
        provider_key=args.provider_key,
        model_name=args.model,
        repetitions=args.repetitions,
        max_prompts=args.max_prompts,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
        gateway_name=None,
        spacing_hours=args.spacing_hours,
        max_retries=args.max_retries,
        attempt_timeout_seconds=args.attempt_timeout_seconds,
        max_output_tokens=args.max_output_tokens,
    )

    print()
    print(f"Wrote    : {args.output_jsonl.resolve()}")
    print(f"Records  : {total_records}")
    print(f"Successes: {success_count}")
    print(f"Failures : {total_records - success_count}")
    return 0 if success_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
