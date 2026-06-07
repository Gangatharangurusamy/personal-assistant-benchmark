import os
import logging
from typing import List, Dict, TypedDict, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from src.adapters.base          import ModelResponse
from src.adapters.hf_adapter    import HFAdapter
from src.adapters.groq_adapter  import GroqAdapter
from src.observability.logger   import ObservabilityLogger
from src.tools.router           import detect_tool
from src.tools.web_search       import web_search
from src.tools.calculator       import calculator
from src.tools.memory_recall    import memory_recall

load_dotenv()

logger  = logging.getLogger(__name__)
_logger = ObservabilityLogger()

SYSTEM_PROMPT = """You are a helpful, harmless, and honest AI assistant.
- Answer questions clearly and concisely.
- If you do not know something, say so honestly.
- Never make up facts.
- Refuse harmful, unethical, or dangerous requests politely.
- Remember context from earlier in the conversation.
- IMPORTANT: When tool results are provided, always trust and
  use the tool results over your training data. Tool results
  are more current and accurate.
"""


# ─────────────────────────────────────────
# STATE — all fields defined upfront
# ─────────────────────────────────────────

class ConversationState(TypedDict):
    user_input:     str
    messages:       List[Dict]
    response:       str
    model_response: Dict
    is_harmful:     bool
    adapter_type:   str
    tool_result:    str
    tool_used:      str


# ─────────────────────────────────────────
# NODES
# ─────────────────────────────────────────

def guardrail_node(state: ConversationState) -> ConversationState:
    logger.info(f"[GUARDRAIL] checking: {state['user_input'][:50]}")

    harmful_keywords = [
        "bomb", "kill", "hack", "malware", "drug synthesis",
        "child abuse", "terrorist", "weapon", "suicide method"
    ]
    is_harmful = any(
        kw in state["user_input"].lower()
        for kw in harmful_keywords
    )
    logger.info(f"[GUARDRAIL] is_harmful={is_harmful}")
    return {**state, "is_harmful": is_harmful}


def tool_node(state: ConversationState) -> ConversationState:
    logger.info(f"[TOOL NODE] detecting tool for: {state['user_input'][:50]}")

    tool = detect_tool(state["user_input"])
    result = ""

    if tool == "web_search":
        result = web_search(state["user_input"])

    elif tool == "calculator":
        result = calculator(state["user_input"])

    elif tool == "memory_recall":
        result = memory_recall(
            state["messages"],
            state["user_input"]
        )

    logger.info(f"[TOOL NODE] tool={tool} result_len={len(result)}")
    return {**state, "tool_result": result, "tool_used": tool}


def llm_node(state: ConversationState) -> ConversationState:
    logger.info(f"[LLM NODE] adapter={state['adapter_type']}")

    if state["adapter_type"] == "hf":
        adapter = HFAdapter(system_prompt=SYSTEM_PROMPT)
    else:
        adapter = GroqAdapter(system_prompt=SYSTEM_PROMPT)

    user_content = state["user_input"]
    if state.get("tool_result"):
        user_content = (
            f"{state['user_input']}\n\n"
            f"[Tool used: {state['tool_used']}]\n"
            f"[Tool result]: {state['tool_result']}\n\n"
            f"Use the above tool result to answer the user."
        )

    messages = state["messages"] + [
        {"role": "user", "content": user_content}
    ]

    result: ModelResponse = adapter.chat(messages)
    logger.info(f"[LLM NODE] response_len={len(result.text)} latency={result.latency_ms}ms")

    _logger.log_call(
        adapter_type  = state["adapter_type"],
        prompt        = state["user_input"],
        response      = result.text,
        tokens_input  = result.tokens_input,
        tokens_output = result.tokens_output,
        latency_ms    = result.latency_ms,
        cost_usd      = result.cost_usd,
        model_name    = result.model_name,
        is_harmful    = False,
    )

    return {
        **state,
        "response": result.text,
        "model_response": {
            "text":          result.text,
            "tokens_input":  result.tokens_input,
            "tokens_output": result.tokens_output,
            "latency_ms":    result.latency_ms,
            "cost_usd":      result.cost_usd,
            "model_name":    result.model_name,
        }
    }


def refuse_node(state: ConversationState) -> ConversationState:
    logger.info("[REFUSE NODE] refusing harmful input")
    refusal = "I'm sorry, I can't help with that. Please ask me something else."

    _logger.log_call(
        adapter_type  = state["adapter_type"],
        prompt        = state["user_input"],
        response      = refusal,
        tokens_input  = 0,
        tokens_output = 0,
        latency_ms    = 0.0,
        cost_usd      = 0.0,
        model_name    = "guardrail",
        is_harmful    = True,
    )

    return {
        **state,
        "response": refusal,
        "model_response": {
            "text":          refusal,
            "tokens_input":  0,
            "tokens_output": 0,
            "latency_ms":    0.0,
            "cost_usd":      0.0,
            "model_name":    "guardrail",
        }
    }


def memory_node(state: ConversationState) -> ConversationState:
    logger.info("[MEMORY NODE] saving turn")
    updated_messages = state["messages"] + [
        {"role": "user",      "content": state["user_input"]},
        {"role": "assistant", "content": state["response"]},
    ]
    return {**state, "messages": updated_messages}


# ─────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────

def route_after_guardrail(
    state: ConversationState,
) -> Literal["tool_node", "refuse_node"]:
    if state["is_harmful"]:
        return "refuse_node"
    return "tool_node"


# ─────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────

def build_conversation_graph():
    graph = StateGraph(ConversationState)

    graph.add_node("guardrail_node", guardrail_node)
    graph.add_node("tool_node",      tool_node)
    graph.add_node("llm_node",       llm_node)
    graph.add_node("refuse_node",    refuse_node)
    graph.add_node("memory_node",    memory_node)

    graph.set_entry_point("guardrail_node")

    graph.add_conditional_edges(
        "guardrail_node",
        route_after_guardrail,
        {
            "tool_node":   "tool_node",
            "refuse_node": "refuse_node",
        }
    )

    graph.add_edge("tool_node",   "llm_node")
    graph.add_edge("llm_node",    "memory_node")
    graph.add_edge("refuse_node", END)
    graph.add_edge("memory_node", END)

    return graph.compile()


# ─────────────────────────────────────────
# ASSISTANT CLASS
# ─────────────────────────────────────────

class ConversationAssistant:

    def __init__(self, adapter_type: str = "groq"):
        self.adapter_type = adapter_type
        self.graph        = build_conversation_graph()
        self.messages     = []

    def chat(self, user_input: str) -> dict:
        initial_state: ConversationState = {
            "user_input":     user_input,
            "messages":       self.messages,
            "response":       "",
            "model_response": {},
            "is_harmful":     False,
            "adapter_type":   self.adapter_type,
            "tool_result":    "",
            "tool_used":      "none",
        }

        result = self.graph.invoke(initial_state)

        self.messages = result["messages"]

        return {
            "response":       result["response"],
            "model_response": result["model_response"],
            "is_harmful":     result["is_harmful"],
            "tool_used":      result["tool_used"],
            "turn_count":     len(self.messages) // 2,
        }

    def reset(self):
        self.messages = []
        logger.info("[ASSISTANT] memory cleared")


# ─────────────────────────────────────────
# PIPELINE EXPORT
# ─────────────────────────────────────────

__all__ = ["ConversationAssistant"]