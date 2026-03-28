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
            "name": "get_stock_price",
            "description": (
                "Get the stock price and performance for a ticker symbol over a given period. "
                "Use for questions about share prices, stock trends, or market performance. "
                "Pass the standard ticker symbol directly (e.g. NVDA, AAPL, TSLA, MC.PA for LVMH). "
                "period: '1d' = today vs yesterday, '7d' = last 7 days, '1mo' = last month, '3mo' = last quarter, '1y' = last 12 months."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. NVDA, AAPL, MC.PA"
                    },
                    "period": {
                        "type": "string",
                        "enum": ["1d", "7d", "1mo", "3mo", "1y"],
                        "description": "Time period for the price history"
                    }
                },
                "required": ["ticker", "period"]
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

# Domains known to publish celebrity death hoaxes or satire as news.
_BLOCKED_DOMAINS = {
    "mediamass.net",
    "huzlers.com",
    "empirenews.net",
    "thedailymash.co.uk",
    "theonion.com",
    "babylonbee.com",
}

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
    if name == "get_stock_price":
        return _get_stock_price(arguments.get("ticker", ""), arguments.get("period", "7d"), log)
    if name == "wikipedia_lookup":
        return _wikipedia_lookup(arguments.get("query", ""), log)
    return f"[unknown tool: {name}]"


def _get_stock_price(ticker: str, period: str, log) -> str:
    log.info(f"[tool] get_stock_price({ticker!r}, {period!r})")
    try:
        import yfinance as yf

        # yfinance period/interval mapping
        _interval = {"1d": "5m", "7d": "1d", "1mo": "1d", "3mo": "1d", "1y": "1wk"}.get(period, "1d")
        hist = yf.Ticker(ticker).history(period=period, interval=_interval)

        if hist.empty:
            return f"No data found for ticker '{ticker}'. Check the symbol (e.g. NVDA, MC.PA)."

        current  = hist["Close"].iloc[-1]
        start    = hist["Close"].iloc[0]
        high     = hist["High"].max()
        low      = hist["Low"].min()
        change   = current - start
        pct      = (change / start) * 100
        date     = hist.index[-1].strftime("%Y-%m-%d")
        sign     = "+" if change >= 0 else ""

        return (
            f"{ticker} — {period}: ${current:.2f} (as of {date})\n"
            f"Change: {sign}{change:.2f} USD ({sign}{pct:.1f}%)\n"
            f"High: ${high:.2f} | Low: ${low:.2f}"
        )
    except Exception as exc:
        log.warning(f"get_stock_price error: {exc}")
        return f"Stock data unavailable for '{ticker}': {exc}"


def _web_search(query: str, log) -> str:
    log.info(f"[tool] web_search({query!r})")
    try:
        from duckduckgo_search import DDGS
        results = None
        for backend in ("lite", "html"):
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=3, backend=backend))
                log.debug(f"web_search backend={backend} ok")
                break
            except Exception as e:
                log.warning(f"web_search backend={backend} failed: {e} — trying next")
        results = [r for r in results if not any(d in r["href"] for d in _BLOCKED_DOMAINS)]
        if not results:
            return "No search results found."
        snippets = [f"{r['title']} [{r['href']}]: {r['body']}" for r in results]
        disclaimer = "[Web results — may contain hoaxes or unverified information. Stay critical.]\n\n"
        return (disclaimer + "\n\n".join(snippets))[:2200]
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
