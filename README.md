---
title: Personal Assistant Benchmark
emoji: 🤖
colorFrom: cyan
colorTo: blue
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# LLM Evaluation Harness

Compare OSS vs Frontier LLMs on hallucination, bias, and safety.

## Models

| Role     | Model                  | Provider              | Cost           |
|----------|------------------------|-----------------------|----------------|
| OSS      | Qwen2.5-0.5B-Instruct  | HuggingFace Inference | Free           |
| Frontier | Llama-3.3-70B          | Groq API              | ~$0.0001/call  |

---

## Architecture

```
User Input
↓
[LangGraph Conversation Pipeline]
├── Guardrail Node   ← blocks harmful inputs
├── Tool Node        ← web search / calculator / memory recall
├── LLM Node         ← calls HF or Groq adapter
├── Memory Node      ← saves conversation history
└── Output

[LangGraph Evaluation Pipeline]
├── Prompt Registry  ← 19 structured test prompts across 5 categories
├── Run on OSS       ← Qwen response
├── Run on Frontier  ← Groq response
├── LLM-as-Judge     ← scores both responses
└── Log to DB + JSONL

[Observability Layer]
├── SQLite DB        ← permanent call storage
├── JSONL logs       ← portable log format
└── app.log          ← terminal/file logger

[Reporting Layer]
├── 4 Matplotlib charts
└── PDF report
```

---

## Project Structure

```
ai_personal_assistants/
├── src/
│   ├── adapters/        ← HF + Groq model wrappers
│   ├── prompts/         ← 19 test prompts across 5 categories
│   ├── pipelines/       ← LangGraph conversation + eval pipelines
│   ├── evaluators/      ← LLM-as-judge + scorer
│   ├── tools/           ← web search, calculator, memory recall
│   ├── observability/   ← SQLite + JSONL logger
│   └── reporting/       ← charts + PDF generator
├── tests/               ← unit + integration tests
├── notebooks/           ← exploratory notebooks
├── logs/                ← all logs, charts, reports
├── config/              ← model + eval config YAML
├── app.py               ← Gradio UI
└── requirements.txt
```

---

## Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/ai_personal_assistants
cd ai_personal_assistants
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
```

Get your keys:
- HF Token → https://huggingface.co/settings/tokens
- Groq Key → https://console.groq.com

### 5. Run the app

```bash
python app.py
```

Open browser at `http://localhost:7860`

---

## Running Evaluation

### Quick test (2 categories)

```bash
python tests/test_eval.py
```

### Full evaluation + PDF report

```bash
python tests/test_generate_report.py
```

Report saved to `logs/evaluation_report.pdf`

---

## Architecture Decisions

### Why LangGraph?
- Clean state management for multi-turn conversations
- Conditional routing (guardrail → LLM or refuse)
- Extensible — add nodes without touching existing code

### Why Groq for Frontier?
- Free tier available
- Fastest inference (~600ms vs ~6000ms for HF)
- Serves Llama-3.3-70B — genuinely frontier quality

### Why Qwen2.5-0.5B for OSS?
- Runs free on HF Inference API
- Small enough for zero-cost deployment
- Still surprisingly capable on factual + safety tasks

### Why LLM-as-Judge?
- More nuanced than keyword matching
- Handles open-ended responses fairly
- Same judge model for both — eliminates bias

---

## Tradeoffs

| Decision                  | Tradeoff                                          |
|---------------------------|---------------------------------------------------|
| Qwen 0.5B as OSS          | Free but weaker on complex reasoning              |
| Groq as Frontier          | Fast + cheap but not truly proprietary            |
| SQLite for logs           | Zero setup but not scalable                       |
| Rule-based guardrails     | Fast but misses edge cases                        |
| LLM-as-judge via Groq     | Good quality but adds latency + cost              |
| Hardcoded model config    | Config YAML created but not yet wired into code. Would improve with PyYAML loader in adapters. |

---

## What I Would Improve With More Time

1. MCP tool calling     — current keyword router is a workaround,
                          real LLM-decided tool calling via MCP
                          would be more accurate and flexible

2. Streaming responses  — stream tokens instead of waiting for
                          full response, better UX

3. Better guardrails    — semantic similarity instead of 
                          keyword matching, catches edge cases

4. Multi-turn memory eval — evaluation pipeline currently scores
                            memory as 0.0, needs conversation
                            history passed between turns

5. Larger OSS model     — Qwen2.5-7B for fairer comparison
                          with 70B frontier model

6. PostgreSQL           — replace SQLite for production scale

7. Async evaluation     — run OSS + Frontier calls in parallel,
                          cuts evaluation time by 50%

8. More eval prompts    — 100+ prompts per category for
                          statistically significant results

9. Human feedback loop  — thumbs up/down in UI feeds
                          back to scorer
---

## Evaluation Results Summary

| Category        | OSS (Qwen)  | Frontier (Groq) |
|-----------------|-------------|-----------------|
| Factual         | 10.0        | 10.0            |
| Hallucination   | 5.0 – 7.5   | 10.0            |
| Bias            | 7.5 – 8.25  | 10.0            |
| Jailbreak       | 7.5 – 10.0  | 10.0            |
| Memory          | 0.0*        | 0.0*            |
| **Overall**     | ~7.0        | ~9.0            |

> *Memory scores 0.0 in automated eval — works correctly in live UI.

---

## Live Demo

[Link after HF Spaces deployment]