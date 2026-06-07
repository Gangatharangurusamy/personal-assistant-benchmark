import logging
from ddgs import DDGS

logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 3) -> str:
    try:
        logger.info(f"[TOOL] web_search: {query}")

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return "No results found."

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"{i}. {r.get('title', '')}\n"
                f"   {r.get('body', '')}\n"
                f"   Source: {r.get('href', '')}"
            )

        output = "\n\n".join(formatted)
        logger.info(f"[TOOL] web_search returned {len(results)} results")
        return output

    except Exception as e:
        logger.error(f"[TOOL] web_search error: {e}")
        return f"Search failed: {str(e)}"