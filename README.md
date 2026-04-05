# GateScope Artifact

This directory is a standalone GateScope reference implementation. It is
smaller than the original research codebase and focuses on the parts that
explain the paper's design, runtime workflow, and algorithms.

GateScope audits third-party LLM gateways exposed through the
OpenAI-compatible `/v1/chat/completions` interface. The goal is to measure
observable consistency gaps and supporting signals across four dimensions:

- response content
- multi-turn conversation
- billing
- latency

This repository snapshot does not ship collected data. Runtime outputs go to
`outputs/` and should remain untracked.

## Layout

- `prompts/`
  The 55 single-turn DCC probes and the standardized 25-turn conversation
  workload.
- `collect/`
  Paper-aligned collectors for official APIs and gateways.
- `features/`
  Appendix-aligned signature extraction for normalized scored CSV files.
- `models/`
  The paper's `10/2` split, one-vs-rest XGBoost training, and evaluation.
- `analysis/`
  Conversation, billing, and latency analysis scripts.
- `configs/`
  Example collection and pricing configs.

## Protocol Summary

### Response content

- Fixed DCC prompt contract:
  system prompt plus a user prompt that requires a single JSON object with
  `knowledge_path` and `final_answer`.
- Probe suite composition:
  55 probes total = AIME 15 + GPQA 10 + Geographic 15 + Factual 15.
- Official baseline collection:
  12 repetitions per probe, temperature `0.7`, and at least 2 hours between
  repetitions of the same probe.
- Gateway single-turn collection:
  5 repetitions per probe.

### Multi-turn conversation

- Standardized 25-turn conversation template in `prompts/memory_stress_test.jsonl`.
- The same workload is used for both conversation evaluation and billing
  analysis.
- Conversation aggregates follow the paper table semantics:
  `T10`, `T24`, `T25`, fingerprint count `FC`, and cache rate `CR(%)`,
  aggregated over 5 runs.

### Billing

- Uses the 25-turn standardized workload.
- Computes
  `C_expected = (n_in - n_cached) * p_in + n_cached * p_cached + n_out * p_out`.
- Gateways without cache discounts are evaluated with `n_cached = 0` via
  `apply_cache_discount: false` in the pricing config.

### Latency

- Uses repeated single-turn collection logs.
- Reports end-to-end request duration including retry/backoff overhead inside a
  request, but excluding the fixed sleep between repetition rounds.
- Outputs `CV`, `min`, `max`, `P50`, `P90`, and `P99`.
- Latency is treated as a complementary signal of backend heterogeneity, not as
  standalone proof of a specific internal cause.

## Classifier Algorithm

The classifier branch starts from a normalized scored CSV rather than raw API
logs. It follows the appendix protocol:

1. Extract the response-signature base features:
   `answer_match`, `answer_position`, `depth`, `mean_step_length`,
   `step_length_var`, `response_length`, `density`, `has_numeric`,
   `has_latex`, `parse_success`, `parse_degree`.
2. For each target model and each base feature, compute target-vs-rest
   contrastive statistics on the same `test_id`:
   mean difference, relative difference, Cohen's d, standard deviation ratio,
   and overlap ratio.
3. Add one ranking feature per base feature: the normalized rank of the target
   model among all official models on that `test_id`.
4. Split official data by `(model_name, test_id)` into 10 training rows and 2
   testing rows.
5. Train 24 one-vs-rest XGBoost classifiers.
6. Apply `RandomOverSampler` with a 20x positive-class target.
7. Use the difficulty-adaptive `scale_pos_weight` policy from the paper.
8. Hold out a validation subset for early stopping and threshold search.
9. Sweep thresholds from `0.10` to `0.95` in steps of `0.05`.

## Quick Start

These commands assume your shell is already inside the `GateScope/` root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Smoke-test the collector

For a smoke test, override the official 2-hour spacing to `0` and cap prompts:

```bash
python3 collect/collect_official.py \
  --config configs/config.example.yaml \
  --prompt-file prompts/AIME_2024.jsonl \
  --model gpt-4o \
  --max-prompts 2 \
  --repetitions 1 \
  --spacing-hours 0
```

For a paper-aligned official run, use the config defaults instead of overriding
`repetitions`, `temperature`, `timeout`, or `spacing`.

### Run the classifier branch

```bash
python3 features/extract_content_features.py \
  --input-csv /path/to/normalized_official.csv \
  --output-csv outputs/processed/official_signature.csv

python3 models/split_data.py \
  --input-csv outputs/processed/official_signature.csv \
  --output-dir outputs/processed/binary_protocol

python3 models/train_xgboost.py \
  --train-csv outputs/processed/binary_protocol/binary_train.csv \
  --model-dir outputs/models/binary_protocol

python3 models/evaluate_official.py \
  --test-csv outputs/processed/binary_protocol/binary_test.csv \
  --model-dir outputs/models/binary_protocol \
  --output-dir outputs/evaluation/official_binary

python3 models/evaluate_gateway.py \
  --input-csv /path/to/gateway_scored.csv \
  --model-dir outputs/models/binary_protocol \
  --output-dir outputs/evaluation/gateway_binary
```

If a local official CSV is missing repetitions for a small number of groups, use
`--drop-incomplete-groups` with `models/split_data.py`. The paper protocol
itself still assumes exactly 12 repetitions per `(model_name, test_id)` group.

### Run the conversation, billing, and latency analyses

```bash
python3 analysis/analyze_conversation.py \
  --input-path /path/to/conversation_progress_dir \
  --output-dir outputs/analysis/conversation

python3 analysis/analyze_billing.py \
  --input-path /path/to/conversation_progress_dir \
  --pricing-json configs/pricing.example.json \
  --output-dir outputs/analysis/billing

python3 analysis/analyze_latency.py \
  --input-path /path/to/collection_jsonl_or_dir \
  --output-dir outputs/analysis/latency
```

## Key Outputs

- `collect/*.py`
  Writes normalized JSONL plus a manifest JSON.
- `analysis/analyze_conversation.py`
  Writes `conversation_runs.csv` and `conversation_aggregate.csv`.
- `analysis/analyze_billing.py`
  Writes `billing_runs.csv` and `billing_aggregate.csv`.
- `analysis/analyze_latency.py`
  Writes flat latency records plus dataset-level and overall summaries.
- `models/evaluate_gateway.py`
  Writes row-level gateway predictions and the claimed-model rate summaries used
  for the response-content tables.

## Publication Checklist

- Replace the placeholder endpoints in `configs/config.example.yaml`.
- Add a repository license before publishing.
- Keep `outputs/` untracked.
- Document any external scorer or normalization step used to produce the input
  CSV for the classifier branch.
