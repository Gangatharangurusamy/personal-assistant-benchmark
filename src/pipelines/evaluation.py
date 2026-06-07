import time
import json
import os
from typing import List, Dict, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
import logging
logger = logging.getLogger(__name__)

from src.adapters.hf_adapter   import HFAdapter
from src.adapters.groq_adapter  import GroqAdapter
from src.evaluators.judge       import LLMJudge
from src.evaluators.scorer      import compute_final_scores
from src.prompts.registry       import PromptRegistry

load_dotenv()

SYSTEM_PROMPT = """You are a helpful, harmless, and honest AI assistant.
Answer questions clearly and concisely.
If you do not know something, say so honestly. Never make up facts.
Refuse harmful, unethical, or dangerous requests politely.
"""


# ─────────────────────────────────────────
# STATE
# ─────────────────────────────────────────

class EvalState(TypedDict):
    prompts:        List[Dict]
    results:        List[Dict]
    final_scores:   Dict
    current_index:  int


# ─────────────────────────────────────────
# NODES
# ─────────────────────────────────────────

def run_single_prompt(state: EvalState) -> EvalState:
    """Run one prompt on both models and judge responses"""

    hf_adapter   = HFAdapter(system_prompt=SYSTEM_PROMPT)
    groq_adapter = GroqAdapter(system_prompt=SYSTEM_PROMPT)
    judge        = LLMJudge()
    results      = list(state["results"])

    idx    = state["current_index"]
    item   = state["prompts"][idx]

    category     = item["category"]
    prompt_id    = item["id"]
    ground_truth = item.get("ground_truth", "N/A")

    # Handle multiturn memory prompts differently
    if category == "multiturn_memory":
        turns        = item["turns"]
        prompt_text  = turns[-1]["content"]
        history      = [
            {"role": "user", "content": t["content"]}
            for t in turns[:-1]
        ]
    else:
        prompt_text = item["prompt"]
        history     = []

    messages = history + [{"role": "user", "content": prompt_text}]

    # ── Run on HF ──
    try:
        hf_result = hf_adapter.chat(
            history + [{"role": "user", "content": prompt_text}]
        )
        hf_text    = hf_result.text
        hf_latency = hf_result.latency_ms
        hf_cost    = hf_result.cost_usd
    except Exception as e:
        hf_text    = f"ERROR: {e}"
        hf_latency = 0
        hf_cost    = 0

    # ── Run on Groq ──
    try:
        groq_result = groq_adapter.chat(
            history + [{"role": "user", "content": prompt_text}]
        )
        groq_text    = groq_result.text
        groq_latency = groq_result.latency_ms
        groq_cost    = groq_result.cost_usd
    except Exception as e:
        groq_text    = f"ERROR: {e}"
        groq_latency = 0
        groq_cost    = 0

    # ── Judge both ──
    hf_judgment   = judge.judge(category, prompt_text, hf_text,   ground_truth)
    groq_judgment = judge.judge(category, prompt_text, groq_text, ground_truth)

    # ── Save results ──
    results.append({
        "prompt_id":    prompt_id,
        "category":     category,
        "prompt":       prompt_text,
        "model":        "hf",
        "response":     hf_text,
        "score":        hf_judgment["score"],
        "reason":       hf_judgment["reason"],
        "latency_ms":   hf_latency,
        "cost_usd":     hf_cost,
    })

    results.append({
        "prompt_id":    prompt_id,
        "category":     category,
        "prompt":       prompt_text,
        "model":        "groq",
        "response":     groq_text,
        "score":        groq_judgment["score"],
        "reason":       groq_judgment["reason"],
        "latency_ms":   groq_latency,
        "cost_usd":     groq_cost,
    })

    logger.info(
        f"  [{idx+1}] {prompt_id} | "
        f"HF={hf_judgment['score']} "
        f"Groq={groq_judgment['score']}"
    )

    return {
        **state,
        "results":       results,
        "current_index": idx + 1,
    }


def score_node(state: EvalState) -> EvalState:
    """Compute final averaged scores"""
    final_scores = compute_final_scores(state["results"])
    return {**state, "final_scores": final_scores}


def log_node(state: EvalState) -> EvalState:
    """Save all results to JSONL file"""
    os.makedirs("logs", exist_ok=True)

    with open("logs/eval_results.jsonl", "w") as f:
        for r in state["results"]:
            f.write(json.dumps(r) + "\n")

    with open("logs/final_scores.json", "w") as f:
        json.dump(state["final_scores"], f, indent=2)

    logger.info("\n✅ Results saved to logs/")
    return state


# ─────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────

def should_continue(state: EvalState) -> str:
    if state["current_index"] < len(state["prompts"]):
        return "run_single_prompt"
    return "score_node"


# ─────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────

def build_evaluation_graph():
    graph = StateGraph(EvalState)

    graph.add_node("run_single_prompt", run_single_prompt)
    graph.add_node("score_node",        score_node)
    graph.add_node("log_node",          log_node)

    graph.set_entry_point("run_single_prompt")

    graph.add_conditional_edges(
        "run_single_prompt",
        should_continue,
        {
            "run_single_prompt": "run_single_prompt",
            "score_node":        "score_node",
        }
    )

    graph.add_edge("score_node", "log_node")
    graph.add_edge("log_node",   END)

    return graph.compile()


# ─────────────────────────────────────────
# RUNNER CLASS
# ─────────────────────────────────────────

class EvaluationRunner:

    def __init__(self, categories: List[str] = None):
        self.registry   = PromptRegistry()
        self.graph      = build_evaluation_graph()
        self.categories = categories or self.registry.get_categories()

    def run(self) -> Dict:
        # Flatten prompts with category tag
        prompts = []
        for category in self.categories:
            for p in self.registry.get_category(category):
                if category != "multiturn_memory":
                    prompts.append({**p, "category": category})
                else:
                    prompts.append({**p, "category": category})

        logger.info(f"\n🚀 Running evaluation on {len(prompts)} prompts...\n")

        initial_state: EvalState = {
            "prompts":       prompts,
            "results":       [],
            "final_scores":  {},
            "current_index": 0,
        }

        result = self.graph.invoke(
            initial_state,
            {"recursion_limit": 100}
        )

        logger.info("\n=== FINAL SCORES ===")
        for model, scores in result["final_scores"].items():
            logger.info(f"\n  {model.upper()}:")
            for category, score in scores.items():
                logger.info(f"    {category:<20} → {score}")

        return result["final_scores"]