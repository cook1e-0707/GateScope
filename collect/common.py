from __future__ import annotations

import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from artifact_utils import read_jsonl, read_yaml, write_json, write_jsonl


DCC_SYSTEM_PROMPT = (
    "You are a meticulous reasoning engine. Your task is to solve multi-step "
    "problems by thinking step-by-step and to clearly articulate your reasoning "
    "process. Your final output must be a single JSON object."
)

DCC_USER_TEMPLATE = """Based on the following question, provide your step-by-step reasoning path and the final answer.

**Question**: {question_text}

**Required Output Format**:
Your entire response must be a single JSON object containing the following two keys:
1. `knowledge_path`: An array of strings. Each string in the array should represent a distinct step in your reasoning process.
2. `final_answer`: A string containing the final answer.

**Example**:
**Question**: What is the highest geographic feature associated with the origin area of the Starbucks corporation?
**Your Output should be**:
{{
  "knowledge_path": [
    "Starbucks originated in Seattle, Washington.",
    "The highest geographic feature in the state of Washington is Mount Rainier."
  ],
  "final_answer": "Mount Rainier"
}}

Now, please apply this reasoning process and format to the following question.

**Question**: {question_text}"""

RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504, 529}
OVERLOAD_KEYWORDS = [
    "overload",
    "overloaded",
    "temporarily unavailable",
    "server error",
    "upstream",
    "connection reset",
    "timed out",
    "timeout",
]
TOKEN_PARAM_HINTS = [
    "max_tokens",
    "max_completion_tokens",
]

MODEL_PARAM_RULES = {
    "o-series": {
        "models": {"o1", "o1-mini", "o3", "o3-mini", "o4-mini"},
        "completion_param": "max_completion_tokens",
        "default_max_output_tokens": 100000,
        "supports_temperature": False,
    },
    "gpt-5": {
        "models": {"gpt-5", "gpt-5-mini", "gpt-5-nano"},
        "completion_param": None,
        "default_max_output_tokens": None,
        "supports_temperature": False,
    },
    "default": {
        "models": set(),
        "completion_param": "max_tokens",
        "default_max_output_tokens": 4096,
        "supports_temperature": True,
    },
}


def load_provider_config(config_path: Path, provider_key: str) -> dict[str, Any]:
    config = read_yaml(config_path)
    providers = config.get("providers", {})
    if provider_key not in providers:
        raise KeyError(f"Provider '{provider_key}' not found in {config_path}")
    provider = providers[provider_key]
    if not isinstance(provider, dict):
        raise ValueError(f"Provider config must be a mapping: {provider_key}")
    return provider


def load_collection_config(config_path: Path) -> dict[str, Any]:
    config = read_yaml(config_path)
    collection = config.get("collection", {})
    if not isinstance(collection, dict):
        raise ValueError(f"collection must be a mapping: {config_path}")
    return collection


def resolve_collection_value(collection: dict[str, Any], key: str, override: Any, default: Any) -> Any:
    if override is not None:
        return override
    if key in collection:
        return collection[key]
    return default


def load_probe_rows(prompt_file: Path) -> list[dict[str, Any]]:
    raw_rows = read_jsonl(prompt_file)
    dataset_name = prompt_file.stem
    probe_rows: list[dict[str, Any]] = []

    for index, row in enumerate(raw_rows, start=1):
        question_text = row.get("question_prompt_implicit") or row.get("prompt")
        if not question_text:
            raise ValueError(f"Prompt row {index} in {prompt_file} has no question text.")

        probe_rows.append(
            {
                "dataset": dataset_name,
                "test_id": row.get("test_id", f"{dataset_name}_test_{index:03d}"),
                "question_text": question_text,
                "expected_answer": row.get("final_answer"),
                "prompt_index": index,
            }
        )

    return probe_rows


def build_dcc_messages(question_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DCC_SYSTEM_PROMPT},
        {"role": "user", "content": DCC_USER_TEMPLATE.format(question_text=question_text)},
    ]


def model_param_profile(model_name: str) -> dict[str, Any]:
    for profile_name, profile in MODEL_PARAM_RULES.items():
        if profile_name == "default":
            continue
        if model_name in profile["models"]:
            return profile
    return MODEL_PARAM_RULES["default"]


def response_text_from_payload(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")).strip())
            elif isinstance(item, str):
                chunks.append(item.strip())
        return "\n".join(part for part in chunks if part)
    return str(content).strip()


def system_fingerprint_from_payload(payload: dict[str, Any]) -> str | None:
    fingerprint = payload.get("system_fingerprint")
    return None if fingerprint is None else str(fingerprint)


def usage_from_payload(payload: dict[str, Any]) -> dict[str, int | None]:
    usage = payload.get("usage", {})
    prompt_details = usage.get("prompt_tokens_details", {})
    cached_tokens = usage.get("cached_tokens")
    if cached_tokens is None and isinstance(prompt_details, dict):
        cached_tokens = prompt_details.get("cached_tokens")
    if cached_tokens is None:
        cached_tokens = usage.get("cache_read_input_tokens")

    def _as_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    return {
        "prompt_tokens": _as_int(usage.get("prompt_tokens", usage.get("input_tokens"))),
        "completion_tokens": _as_int(usage.get("completion_tokens", usage.get("output_tokens"))),
        "total_tokens": _as_int(usage.get("total_tokens")),
        "cached_tokens": _as_int(cached_tokens) or 0,
    }


def _request_endpoint(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def _build_request_body(
    *,
    model_name: str,
    question_text: str,
    temperature: float,
    max_output_tokens: int | None,
    token_param_name: str | None,
) -> dict[str, Any]:
    profile = model_param_profile(model_name)
    body: dict[str, Any] = {
        "model": model_name,
        "messages": build_dcc_messages(question_text),
    }
    if profile["supports_temperature"]:
        body["temperature"] = temperature
    if max_output_tokens is not None and token_param_name:
        body[token_param_name] = int(max_output_tokens)
    return body


def _decode_http_error(exc: error.HTTPError) -> tuple[int, str]:
    raw_body = exc.read()
    if isinstance(raw_body, bytes):
        try:
            body_text = raw_body.decode("utf-8")
        except UnicodeDecodeError:
            body_text = raw_body.decode("utf-8", errors="replace")
    else:
        body_text = ""
    return int(exc.code), body_text


def _retryable_message(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in OVERLOAD_KEYWORDS)


def _token_param_switch(status_code: int, body_text: str, current_param_name: str | None) -> str | None:
    if status_code != 400:
        return None
    lowered = body_text.lower()
    if not any(hint in lowered for hint in TOKEN_PARAM_HINTS):
        return None
    if current_param_name == "max_tokens" and "max_completion_tokens" in lowered:
        return "max_completion_tokens"
    if current_param_name == "max_completion_tokens" and "max_tokens" in lowered:
        return "max_tokens"
    if current_param_name is None and "max_completion_tokens" in lowered:
        return "max_completion_tokens"
    if current_param_name is None and "max_tokens" in lowered:
        return "max_tokens"
    return None


def _backoff_seconds(attempt_number: int) -> float:
    return min(2.0 ** max(attempt_number - 1, 0), 60.0)


def send_chat_completion_with_protocol(
    *,
    base_url: str,
    api_key: str,
    model_name: str,
    question_text: str,
    temperature: float,
    total_timeout_seconds: float,
    max_retries: int,
    attempt_timeout_seconds: float | None,
    max_output_tokens: int | None,
) -> dict[str, Any]:
    endpoint = _request_endpoint(base_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    profile = model_param_profile(model_name)
    token_param_name = profile["completion_param"]
    if max_output_tokens is None:
        token_param_name = None

    started_at = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    deadline = wall_start + total_timeout_seconds

    attempt_logs: list[dict[str, Any]] = []
    payload: dict[str, Any] | None = None
    last_status: int | None = None
    last_error_message: str | None = None

    max_attempts = max_retries + 1
    for attempt_number in range(1, max_attempts + 1):
        remaining_seconds = deadline - time.perf_counter()
        if remaining_seconds <= 0:
            last_error_message = "Total timeout exceeded before next attempt."
            break

        timeout_for_attempt = remaining_seconds
        if attempt_timeout_seconds is not None:
            timeout_for_attempt = min(timeout_for_attempt, attempt_timeout_seconds)

        body = _build_request_body(
            model_name=model_name,
            question_text=question_text,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            token_param_name=token_param_name,
        )
        encoded = json.dumps(body).encode("utf-8")
        http_request = request.Request(endpoint, data=encoded, headers=headers, method="POST")

        attempt_started_at = datetime.now(timezone.utc).isoformat()
        attempt_start = time.perf_counter()
        retryable = False
        backoff_seconds = 0.0
        error_message = None
        status_code = None

        try:
            with request.urlopen(http_request, timeout=timeout_for_attempt) as response:
                payload = json.loads(response.read().decode("utf-8"))
            duration_seconds = time.perf_counter() - attempt_start
            attempt_logs.append(
                {
                    "attempt_number": attempt_number,
                    "started_at": attempt_started_at,
                    "duration_seconds": float(duration_seconds),
                    "status": "success",
                    "http_status": 200,
                    "retryable": False,
                    "backoff_seconds": 0.0,
                    "token_param_name": token_param_name,
                }
            )
            break
        except error.HTTPError as exc:
            status_code, body_text = _decode_http_error(exc)
            duration_seconds = time.perf_counter() - attempt_start
            switch_param = _token_param_switch(status_code, body_text, token_param_name)
            if switch_param and switch_param != token_param_name:
                token_param_name = switch_param
                retryable = True
                error_message = f"HTTP {status_code}: token parameter compatibility adjustment"
            else:
                retryable = status_code in RETRYABLE_HTTP_STATUS or _retryable_message(body_text)
                body_summary = body_text.strip().replace("\n", " ")
                error_message = f"HTTP {status_code}: {body_summary[:240]}"
            last_status = status_code
            last_error_message = error_message
        except (error.URLError, TimeoutError, socket.timeout) as exc:
            duration_seconds = time.perf_counter() - attempt_start
            retryable = True
            error_message = f"{type(exc).__name__}: {exc}"
            last_error_message = error_message
        except Exception as exc:  # noqa: BLE001
            duration_seconds = time.perf_counter() - attempt_start
            error_message = f"{type(exc).__name__}: {exc}"
            retryable = _retryable_message(error_message)
            last_error_message = error_message

        if payload is None:
            if attempt_number < max_attempts and retryable:
                remaining_after_attempt = deadline - time.perf_counter()
                if remaining_after_attempt > 0:
                    backoff_seconds = min(_backoff_seconds(attempt_number), max(remaining_after_attempt, 0.0))
                    if backoff_seconds > 0:
                        time.sleep(backoff_seconds)

            attempt_logs.append(
                {
                    "attempt_number": attempt_number,
                    "started_at": attempt_started_at,
                    "duration_seconds": float(duration_seconds),
                    "status": "error",
                    "http_status": status_code,
                    "retryable": bool(retryable),
                    "error_message": error_message,
                    "backoff_seconds": float(backoff_seconds),
                    "token_param_name": token_param_name,
                }
            )

            if not retryable or attempt_number >= max_attempts:
                break

    finished_at = datetime.now(timezone.utc).isoformat()
    request_duration_seconds = time.perf_counter() - wall_start

    if payload is None:
        return {
            "success": False,
            "started_at": started_at,
            "finished_at": finished_at,
            "request_duration_seconds": float(request_duration_seconds),
            "attempt_count": len(attempt_logs),
            "attempt_logs": attempt_logs,
            "http_status": last_status,
            "error_message": last_error_message,
            "response_text": "",
            "system_fingerprint": None,
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "cached_tokens": 0,
            },
            "provider_response_id": None,
        }

    return {
        "success": True,
        "started_at": started_at,
        "finished_at": finished_at,
        "request_duration_seconds": float(request_duration_seconds),
        "attempt_count": len(attempt_logs),
        "attempt_logs": attempt_logs,
        "http_status": 200,
        "error_message": None,
        "response_text": response_text_from_payload(payload),
        "system_fingerprint": system_fingerprint_from_payload(payload),
        "usage": usage_from_payload(payload),
        "provider_response_id": payload.get("id"),
    }


def _write_manifest(output_jsonl: Path, manifest: dict[str, Any]) -> None:
    manifest_path = output_jsonl.with_suffix(".manifest.json")
    write_json(manifest_path, manifest)


def collect_responses(
    *,
    prompt_file: Path,
    output_jsonl: Path,
    config_path: Path,
    provider_key: str,
    model_name: str,
    repetitions: int | None,
    max_prompts: int | None,
    temperature: float | None,
    timeout_seconds: float | None,
    gateway_name: str | None,
    spacing_hours: float | None = None,
    max_retries: int | None = None,
    attempt_timeout_seconds: float | None = None,
    max_output_tokens: int | None = None,
) -> tuple[int, int]:
    provider = load_provider_config(config_path, provider_key)
    collection = load_collection_config(config_path)

    api_key_env = provider.get("api_key_env")
    base_url = provider.get("base_url")
    if not api_key_env or not base_url:
        raise ValueError(f"Provider '{provider_key}' must define base_url and api_key_env.")

    api_key = os.environ.get(str(api_key_env))
    if not api_key:
        raise EnvironmentError(f"Missing API key. Export {api_key_env} before running the collector.")

    default_repetitions_key = "gateway_repetitions_per_probe" if gateway_name else "repetitions_per_probe"
    default_spacing_key = "gateway_spacing_hours" if gateway_name else "official_spacing_hours"

    resolved_repetitions = int(resolve_collection_value(collection, default_repetitions_key, repetitions, 5))
    resolved_temperature = float(resolve_collection_value(collection, "temperature", temperature, 0.7))
    resolved_timeout_seconds = float(resolve_collection_value(collection, "total_timeout_seconds", timeout_seconds, 900.0))
    resolved_spacing_hours = float(resolve_collection_value(collection, default_spacing_key, spacing_hours, 0.0))
    resolved_max_retries = int(resolve_collection_value(collection, "max_retries", max_retries, 15))
    resolved_attempt_timeout_seconds = resolve_collection_value(
        collection,
        "attempt_timeout_seconds",
        attempt_timeout_seconds,
        None,
    )
    if resolved_attempt_timeout_seconds is not None:
        resolved_attempt_timeout_seconds = float(resolved_attempt_timeout_seconds)
    resolved_max_output_tokens = resolve_collection_value(collection, "max_output_tokens", max_output_tokens, None)
    if resolved_max_output_tokens is None:
        resolved_max_output_tokens = model_param_profile(model_name)["default_max_output_tokens"]
    if resolved_max_output_tokens is not None:
        resolved_max_output_tokens = int(resolved_max_output_tokens)

    probes = load_probe_rows(prompt_file)
    if max_prompts is not None:
        probes = probes[:max_prompts]

    records: list[dict[str, Any]] = []
    total_rounds = resolved_repetitions
    manifest = {
        "protocol": "paper_dcc_sampling_v1",
        "provider_name": provider_key,
        "gateway_name": gateway_name,
        "model_name": model_name,
        "prompt_file": str(prompt_file),
        "output_jsonl": str(output_jsonl),
        "records_expected": int(len(probes) * total_rounds),
        "probes": int(len(probes)),
        "repetitions_per_probe": resolved_repetitions,
        "temperature": resolved_temperature,
        "total_timeout_seconds": resolved_timeout_seconds,
        "max_retries": resolved_max_retries,
        "attempt_timeout_seconds": resolved_attempt_timeout_seconds,
        "spacing_hours_between_repetitions_of_same_probe": resolved_spacing_hours,
        "max_output_tokens": resolved_max_output_tokens,
    }

    for round_index in range(1, total_rounds + 1):
        round_started = time.time()
        print(f"[round {round_index}/{total_rounds}] {len(probes)} probes")
        for probe in probes:
            protocol_result = send_chat_completion_with_protocol(
                base_url=str(base_url),
                api_key=api_key,
                model_name=model_name,
                question_text=str(probe["question_text"]),
                temperature=resolved_temperature,
                total_timeout_seconds=resolved_timeout_seconds,
                max_retries=resolved_max_retries,
                attempt_timeout_seconds=resolved_attempt_timeout_seconds,
                max_output_tokens=resolved_max_output_tokens,
            )
            usage = protocol_result["usage"]
            records.append(
                {
                    "provider_name": provider_key,
                    "gateway_name": gateway_name,
                    "model_name": model_name,
                    "dataset": probe["dataset"],
                    "test_id": probe["test_id"],
                    "prompt_index": probe["prompt_index"],
                    "repetition": round_index,
                    "timestamp": protocol_result["started_at"],
                    "started_at": protocol_result["started_at"],
                    "finished_at": protocol_result["finished_at"],
                    "question_text": probe["question_text"],
                    "expected_answer": probe["expected_answer"],
                    "response_text": protocol_result["response_text"],
                    "response_length": len(protocol_result["response_text"]),
                    "request_duration_seconds": protocol_result["request_duration_seconds"],
                    "latency_seconds": protocol_result["request_duration_seconds"],
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "cached_tokens": usage["cached_tokens"],
                    "system_fingerprint": protocol_result["system_fingerprint"],
                    "provider_response_id": protocol_result["provider_response_id"],
                    "http_status": protocol_result["http_status"],
                    "success": protocol_result["success"],
                    "attempt_count": protocol_result["attempt_count"],
                    "attempt_logs": protocol_result["attempt_logs"],
                    "error_message": protocol_result["error_message"],
                    "protocol_name": "paper_dcc_sampling_v1",
                    "protocol_schema": "knowledge_path + final_answer JSON object",
                }
            )

            status = "ok" if protocol_result["success"] else "error"
            print(
                f"[{status}] {probe['test_id']} rep={round_index} "
                f"attempts={protocol_result['attempt_count']} "
                f"duration={protocol_result['request_duration_seconds']:.2f}s"
            )

        if round_index < total_rounds and resolved_spacing_hours > 0:
            elapsed_round_seconds = time.time() - round_started
            target_round_seconds = resolved_spacing_hours * 3600.0
            sleep_seconds = max(target_round_seconds - elapsed_round_seconds, 0.0)
            if sleep_seconds > 0:
                print(f"Sleeping {sleep_seconds / 3600.0:.2f}h before next repetition round.")
                time.sleep(sleep_seconds)

    write_jsonl(output_jsonl, records)
    manifest["records_written"] = int(len(records))
    manifest["success_count"] = int(sum(1 for record in records if record["success"]))
    manifest["failure_count"] = int(len(records) - manifest["success_count"])
    _write_manifest(output_jsonl, manifest)

    success_count = sum(1 for record in records if record["success"])
    return len(records), success_count
