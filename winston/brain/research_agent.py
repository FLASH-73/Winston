"""Lightweight research agent — Sonnet + text tools, no computer use.

Runs in a background thread so the user can keep chatting.
Uses standard messages API (NOT the beta computer_use endpoint).
"""

import logging
import threading
import time
from typing import Callable, Optional
from uuid import uuid4

import httpx

logger = logging.getLogger("winston.research")

# Tool definitions for the Anthropic API
RESEARCH_TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web using DuckDuckGo. Returns top results with titles, URLs, and snippets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Max results (default 5)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_webpage",
        "description": "Fetch a webpage and return its text content (HTML stripped).",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "github_read_file",
        "description": "Read a file from a GitHub repository (public repos only).",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository in owner/name format"},
                "path": {"type": "string", "description": "File path within the repo"},
                "branch": {"type": "string", "description": "Branch name (default: main)"},
            },
            "required": ["repo", "path"],
        },
    },
    {
        "name": "github_list_repo_files",
        "description": "List all files in a GitHub repository to understand its structure.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository in owner/name format"},
                "branch": {"type": "string", "description": "Branch name (default: main)"},
            },
            "required": ["repo"],
        },
    },
    {
        "name": "github_list_repos",
        "description": "List GitHub repositories for a user or organization.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "GitHub username or org"},
            },
            "required": ["owner"],
        },
    },
]

# Tool execution functions

def _web_search(inputs: dict) -> str:
    """Search the web using DuckDuckGo (no API key needed)."""
    query = inputs.get("query", "")
    max_results = inputs.get("max_results", 5)
    if not query:
        return "Error: query is required"
    try:
        from ddgs import DDGS

        results = DDGS().text(query, max_results=max_results)
        if not results:
            return f"No results found for: {query}"
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            body = r.get("body", "")
            href = r.get("href", "")
            lines.append(f"{i}. {title}\n   {href}\n   {body}")
        return "\n\n".join(lines)
    except ImportError:
        return "Error: ddgs package not installed. Run: pip install ddgs"
    except Exception as e:
        return f"Web search error: {e}"


def _fetch_webpage(inputs: dict) -> str:
    """Fetch a webpage and return its text content (HTML stripped)."""
    import re as re_mod

    url = inputs["url"]
    try:
        resp = httpx.get(
            url,
            follow_redirects=True,
            timeout=15.0,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Winston/1.0)"},
        )
        resp.raise_for_status()
        html = resp.text
        html = re_mod.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re_mod.DOTALL | re_mod.IGNORECASE)
        text = re_mod.sub(r"<[^>]+>", " ", html)
        text = re_mod.sub(r"\s+", " ", text).strip()
        return text[:4000] if text else "(empty page)"
    except httpx.HTTPStatusError as e:
        return f"HTTP error {e.response.status_code} fetching {url}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


def _github_read_file(inputs: dict) -> str:
    """Read a file from a public GitHub repo via raw.githubusercontent.com."""
    repo = inputs["repo"]
    path = inputs["path"]
    branch = inputs.get("branch", "main")
    url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
    try:
        resp = httpx.get(url, timeout=10, follow_redirects=True)
        resp.raise_for_status()
        text = resp.text
        return text[:4000] if text else "(empty file)"
    except httpx.HTTPStatusError as e:
        return f"HTTP error {e.response.status_code} reading {repo}/{path}"
    except Exception as e:
        return f"Error reading file: {e}"


def _github_list_repo_files(inputs: dict) -> str:
    """List files in a GitHub repo using the Git Trees API."""
    repo = inputs["repo"]
    branch = inputs.get("branch", "main")
    url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
    try:
        resp = httpx.get(url, timeout=10, headers={"Accept": "application/vnd.github.v3+json"})
        resp.raise_for_status()
        tree = resp.json().get("tree", [])
        lines = []
        for item in tree:
            if item.get("type") == "blob":
                lines.append(item["path"])
        return "\n".join(lines) if lines else "(empty repository)"
    except httpx.HTTPStatusError as e:
        return f"HTTP error {e.response.status_code} listing {repo}"
    except Exception as e:
        return f"Error listing repo files: {e}"


def _github_list_repos(inputs: dict) -> str:
    """List public repos for a GitHub user/org."""
    owner = inputs["owner"]
    url = f"https://api.github.com/users/{owner}/repos?sort=updated&per_page=30"
    try:
        resp = httpx.get(url, timeout=10, headers={"Accept": "application/vnd.github.v3+json"})
        resp.raise_for_status()
        repos = resp.json()
        lines = []
        for r in repos:
            desc = r.get("description", "") or ""
            lines.append(f"- {r['name']}: {desc[:80]}")
        return "\n".join(lines) if lines else f"No public repos found for {owner}"
    except httpx.HTTPStatusError as e:
        return f"HTTP error {e.response.status_code} listing repos for {owner}"
    except Exception as e:
        return f"Error listing repos: {e}"


# Tool name → executor mapping
_TOOL_EXECUTORS = {
    "web_search": _web_search,
    "fetch_webpage": _fetch_webpage,
    "github_read_file": _github_read_file,
    "github_list_repo_files": _github_list_repo_files,
    "github_list_repos": _github_list_repos,
}

RESEARCH_SYSTEM_PROMPT = """You are a research assistant. Complete the research task efficiently.
You have access to web search, webpage reading, and GitHub tools (for public repos).
Be thorough but concise. Summarize your findings clearly.
Do NOT use more than 8 tool calls. Focus on the most relevant information.
When done, write a clear summary of what you found."""


class ResearchAgent:
    """Lightweight research agent — runs Sonnet in a background thread with text-only tools."""

    def __init__(self, client, cost_tracker):
        """
        Args:
            client: Anthropic client instance (already authenticated).
            cost_tracker: CostTracker for recording token usage.
        """
        self._client = client
        self._cost_tracker = cost_tracker

    def run_research(
        self,
        task: str,
        context: str,
        callback: Callable[[str], None],
    ) -> str:
        """Launch research in a background thread. Returns task ID."""
        task_id = uuid4().hex[:8]
        thread = threading.Thread(
            target=self._execute_research,
            args=(task, context, callback),
            daemon=True,
            name=f"research-{task_id}",
        )
        thread.start()
        logger.info("[research] Started task %s: '%s'", task_id, task[:100])
        return task_id

    def _execute_research(
        self,
        task: str,
        context: str,
        callback: Callable[[str], None],
    ):
        """Agentic loop: Sonnet + tools, max iterations, no computer use."""
        from config import RESEARCH_MAX_ITERATIONS, RESEARCH_MAX_TOKENS, RESEARCH_MODEL

        full_task = task
        if context:
            full_task = f"{task}\n\nContext:\n{context}"

        messages = [{"role": "user", "content": full_task}]

        try:
            for iteration in range(RESEARCH_MAX_ITERATIONS):
                logger.info("[research] Iteration %d/%d", iteration + 1, RESEARCH_MAX_ITERATIONS)

                response = self._client.messages.create(
                    model=RESEARCH_MODEL,
                    system=RESEARCH_SYSTEM_PROMPT,
                    messages=messages,
                    tools=RESEARCH_TOOLS,
                    max_tokens=RESEARCH_MAX_TOKENS,
                )

                # Track cost
                usage = response.usage
                self._cost_tracker.record(
                    "smart",
                    getattr(usage, "input_tokens", 0),
                    getattr(usage, "output_tokens", 0),
                )

                # Check if done (no more tool calls)
                if response.stop_reason != "tool_use":
                    text = self._extract_text(response)
                    logger.info("[research] Complete in %d iterations", iteration + 1)
                    callback(text or "Research completed but no summary was generated.")
                    return

                # Execute tool calls
                messages.append({"role": "assistant", "content": response.content})
                tool_results = self._execute_tools(response.content)
                messages.append({"role": "user", "content": tool_results})

            # Hit max iterations — extract whatever we have
            logger.warning("[research] Hit max iterations (%d)", RESEARCH_MAX_ITERATIONS)
            callback("Research hit iteration limit. Partial results may have been gathered.")

        except Exception as e:
            logger.error("[research] Failed: %s", e, exc_info=True)
            callback(f"Research failed: {e}")

    def _execute_tools(self, content) -> list:
        """Execute tool calls from response content, return tool_result blocks."""
        results = []
        for block in content:
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_input = block.input
            executor = _TOOL_EXECUTORS.get(tool_name)

            if executor is None:
                result_text = f"Unknown tool: {tool_name}"
            else:
                try:
                    logger.info("[research] Tool: %s(%s)", tool_name, _summarize_inputs(tool_input))
                    result_text = executor(tool_input)
                except Exception as e:
                    result_text = f"Tool error: {e}"
                    logger.error("[research] Tool %s failed: %s", tool_name, e)

            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_text,
            })

        return results

    @staticmethod
    def _extract_text(response) -> Optional[str]:
        """Extract text from response content blocks."""
        texts = []
        for block in response.content:
            if hasattr(block, "text") and block.text:
                texts.append(block.text)
        return " ".join(texts).strip() if texts else None


def _summarize_inputs(inputs: dict) -> str:
    """Short summary of tool inputs for logging."""
    parts = []
    for k, v in inputs.items():
        sv = str(v)
        if len(sv) > 50:
            sv = sv[:50] + "..."
        parts.append(f"{k}={sv}")
    return ", ".join(parts)
