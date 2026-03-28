"""
Phase 4 — tool definitions and execution (web grounding).

Tools available to the LM:
  - web_search      : DuckDuckGo text search (no API key)
  - wikipedia_lookup: Wikipedia summary

Usage (from lm.py):
    import tools
    tools.DEFINITIONS  # pass to Ollama /api/chat as "tools"
    tools.execute(name, arguments, log)  # run a tool call
"""

# ---------------------------------------------------------------------------
# Tool definitions (Ollama / OpenAI function-calling format)
# ---------------------------------------------------------------------------

DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current or recent information: breaking news, "
                "current events, prices, sports results, recent facts about people or places. "
                "Use when the answer requires up-to-date data not available in training data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query — use the most relevant language"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_lookup",
            "description": (
                "Look up encyclopedic or background information about a topic, person, "
                "place, or concept on Wikipedia. Use for biographies, history, science, "
                "geography, and general factual knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic or entity name to look up"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def execute(name: str, arguments: dict | str, log) -> str:
    """Dispatch a tool call and return a string result for the LM."""
    import json as _json
    if isinstance(arguments, str):
        try:
            arguments = _json.loads(arguments)
        except Exception:
            arguments = {}

    if name == "web_search":
        return _web_search(arguments.get("query", ""), log)
    if name == "wikipedia_lookup":
        return _wikipedia_lookup(arguments.get("query", ""), log)
    return f"[unknown tool: {name}]"


def _web_search(query: str, log) -> str:
    log.info(f"[tool] web_search({query!r})")
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No search results found."
        snippets = [f"{r['title']}: {r['body']}" for r in results]
        return "\n\n".join(snippets)[:2000]
    except Exception as exc:
        log.warning(f"web_search error: {exc}")
        return f"Search unavailable: {exc}"


def _wikipedia_lookup(query: str, log) -> str:
    log.info(f"[tool] wikipedia_lookup({query!r})")
    try:
        import wikipedia
        try:
            return wikipedia.summary(query, sentences=5, auto_suggest=True)[:1500]
        except wikipedia.exceptions.DisambiguationError as e:
            return wikipedia.summary(e.options[0], sentences=5)[:1500]
        except wikipedia.exceptions.PageError:
            return f"No Wikipedia article found for '{query}'."
    except Exception as exc:
        log.warning(f"wikipedia_lookup error: {exc}")
        return f"Lookup unavailable: {exc}"
