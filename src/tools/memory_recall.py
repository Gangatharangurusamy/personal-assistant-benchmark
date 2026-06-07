import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def memory_recall(messages: List[Dict], query: str) -> str:
    """
    Search through conversation history for relevant context.
    Returns matching turns as formatted string.
    """
    try:
        logger.info(f"[TOOL] memory_recall: {query}")

        if not messages:
            return "No conversation history found."

        query_lower = query.lower()

        # Extract keywords from query
        stop_words = {
            "what", "is", "my", "the", "a", "an",
            "did", "i", "say", "tell", "me", "about",
            "do", "you", "remember", "know"
        }
        keywords = [
            w for w in query_lower.split()
            if w not in stop_words and len(w) > 2
        ]

        if not keywords:
            # Return last 3 turns if no keywords
            recent = messages[-6:] if len(messages) > 6 else messages
            return _format_messages(recent)

        # Search for keyword matches in history
        matched = []
        for msg in messages:
            content = msg.get("content", "").lower()
            if any(kw in content for kw in keywords):
                matched.append(msg)

        if not matched:
            return "No relevant information found in conversation history."

        logger.info(f"[TOOL] memory_recall found {len(matched)} matches")
        return _format_messages(matched)

    except Exception as e:
        logger.error(f"[TOOL] memory_recall error: {e}")
        return f"Memory recall failed: {str(e)}"


def _format_messages(messages: List[Dict]) -> str:
    formatted = []
    for msg in messages:
        role    = msg.get("role", "unknown").upper()
        content = msg.get("content", "")[:200]
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)