import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


# Keywords that trigger each tool
TOOL_TRIGGERS = {
    "web_search": [
        "search", "latest", "current", "today", "news",
        "who is", "what is happening", "recent",
        "2024", "2025", "2026", "now", "right now",
        "price of", "weather", "score"
    ],
    "calculator": [
        "calculate", "compute",          # ← removed "what is"
        "how much is",
        "percent", "%", "sqrt",
        "square root", "multiply",
        "divide", "sum", "total", "average"
    ],
    "memory_recall": [
        "remember", "what did i say", "earlier", "before",
        "my name", "i told you", "i said", "what was",
        "recall", "forgot", "conversation"
    ]
}
def detect_tool(user_input: str) -> str:
    """
    Detect which tool (if any) to use based on user input.
    Returns: "web_search" | "calculator" | "memory_recall" | "none"
    """
    user_lower = user_input.lower()

    scores = {"web_search": 0, "calculator": 0, "memory_recall": 0}

    for tool, triggers in TOOL_TRIGGERS.items():
        for trigger in triggers:
            if trigger in user_lower:
                scores[tool] += 1

    best_tool  = max(scores, key=scores.get)
    best_score = scores[best_tool]

    if best_score == 0:
        logger.info("[ROUTER] No tool needed — direct LLM answer")
        return "none"

    logger.info(f"[ROUTER] Tool selected: {best_tool} (score={best_score})")
    return best_tool