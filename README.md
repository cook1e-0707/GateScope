# GateScope

GateScope is a lightweight reference implementation of our auditing framework
for third-party LLM gateways exposed through the OpenAI-compatible
`/v1/chat/completions` API.

It covers four dimensions:

- response content
- multi-turn conversation
- billing
- latency

This repository does not ship collected data. Runtime outputs are written to
`outputs/`.

## Repository Layout

- `prompts/`: single-turn probes and the 25-turn conversation workload
- `collect/`: official API and gateway collectors
- `features/`: response-signature extraction
- `models/`: data split, XGBoost training, and evaluation
- `analysis/`: conversation, billing, and latency analysis
- `configs/`: example configs

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Collect responses

```bash
python3 collect/collect_official.py --config configs/config.example.yaml
python3 collect/collect_gateway.py --config configs/config.example.yaml
```

### 2. Build response-content signatures

```bash
python3 features/extract_content_features.py \
  --input-csv /path/to/official.csv \
  --output-csv outputs/processed/official_signature.csv
```

### 3. Train and evaluate classifiers

```bash
python3 models/split_data.py \
  --input-csv outputs/processed/official_signature.csv \
  --output-dir outputs/processed/binary_protocol

python3 models/train_xgboost.py \
  --train-csv outputs/processed/binary_protocol/binary_train.csv \
  --model-dir outputs/models/binary_protocol

python3 models/evaluate_official.py \
  --test-csv outputs/processed/binary_protocol/binary_test.csv \
  --model-dir outputs/models/binary_protocol \
  --output-dir outputs/evaluation/official

python3 models/evaluate_gateway.py \
  --input-csv /path/to/gateway.csv \
  --model-dir outputs/models/binary_protocol \
  --output-dir outputs/evaluation/gateway
```

### 4. Run system analyses

```bash
python3 analysis/analyze_conversation.py \
  --input-path /path/to/conversation_logs \
  --output-dir outputs/analysis/conversation

python3 analysis/analyze_billing.py \
  --input-path /path/to/conversation_logs \
  --pricing-json configs/pricing.example.json \
  --output-dir outputs/analysis/billing

python3 analysis/analyze_latency.py \
  --input-path /path/to/collection_logs \
  --output-dir outputs/analysis/latency
```

## Outputs

- normalized collection logs
- response-content classification results
- conversation summaries
- billing summaries
- latency summaries
