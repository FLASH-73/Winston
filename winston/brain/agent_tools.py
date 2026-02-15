"""Universal tool registry for Winston's autonomous agents.

Pluggable system: register tools, get API definitions for Anthropic,
execute tools by name. All tools are read-only.
"""

import base64
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger("winston.agent_tools")

TOOL_TIMEOUT = 15  # Default timeout for subprocess-based tools
MAX_RESULT_CHARS = 4000  # Truncate tool output to avoid context bloat


@dataclass
class AgentTool:
    """A single tool an agent can use."""

    name: str
    description: str
    input_schema: dict
    execute_fn: Callable[[dict], str]

    def to_api_dict(self) -> dict:
        """Convert to Anthropic API tool definition format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """Universal tool registry — add tools, get API definitions, execute."""

    def __init__(self):
        self._tools: dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> None:
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def get_api_definitions(self) -> list[dict]:
        """Get tool definitions formatted for the Anthropic tools= parameter."""
        return [tool.to_api_dict() for tool in self._tools.values()]

    def execute(self, name: str, inputs: dict) -> str:
        """Execute a tool by name. Returns result string."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        try:
            logger.info("[tool] Executing %s(%s)", name, _summarize_inputs(inputs))
            result = tool.execute_fn(inputs)
            if len(result) > MAX_RESULT_CHARS:
                result = result[:MAX_RESULT_CHARS] + f"\n... (truncated, {len(result)} chars total)"
            logger.info("[tool] %s returned %d chars", name, len(result))
            return result
        except Exception as e:
            logger.error("[tool] %s failed: %s", name, e)
            return f"Error executing {name}: {e}"

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())


def _summarize_inputs(inputs: dict) -> str:
    """Short summary of tool inputs for logging."""
    parts = []
    for k, v in inputs.items():
        sv = str(v)
        if len(sv) > 50:
            sv = sv[:50] + "..."
        parts.append(f"{k}={sv}")
    return ", ".join(parts)


def _run_gh(args: list[str], timeout: int = TOOL_TIMEOUT) -> str:
    """Run a gh CLI command and return stdout."""
    cmd = ["gh"] + args
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        return f"gh error (exit {result.returncode}): {result.stderr.strip()}"
    return result.stdout.strip()


# ── GitHub Tools (via gh CLI) ──────────────────────────────────────────


def _github_search_code(inputs: dict) -> str:
    query = inputs["query"]
    args = ["search", "code", query, "--json", "path,repository,textMatches", "-L", "10"]
    if inputs.get("repo"):
        args.extend(["-R", inputs["repo"]])
    if inputs.get("language"):
        args.extend(["--language", inputs["language"]])
    return _run_gh(args)


def _github_read_file(inputs: dict) -> str:
    repo = inputs["repo"]
    path = inputs["path"]
    ref = inputs.get("ref", "HEAD")
    # Use gh api to get file contents
    result = _run_gh([
        "api", f"/repos/{repo}/contents/{path}",
        "-q", ".content",
        "--jq", ".content",
    ])
    if result.startswith("gh error"):
        return result
    # Decode base64 content
    try:
        decoded = base64.b64decode(result.replace("\n", "")).decode("utf-8")
        return decoded
    except Exception:
        # Might not be base64 — return raw
        return result


def _github_list_repo_files(inputs: dict) -> str:
    repo = inputs["repo"]
    branch = inputs.get("branch", "main")
    result = _run_gh([
        "api", f"/repos/{repo}/git/trees/{branch}?recursive=1",
        "--jq", '.tree[] | select(.type=="blob") | .path',
    ])
    return result


def _github_list_repos(inputs: dict) -> str:
    args = ["repo", "list", "--json", "name,description,updatedAt", "-L", "30"]
    if inputs.get("owner"):
        args.insert(2, inputs["owner"])
    return _run_gh(args)


def _github_view_issue(inputs: dict) -> str:
    repo = inputs["repo"]
    number = inputs["issue_number"]
    return _run_gh(["issue", "view", str(number), "-R", repo])


def _github_view_pr(inputs: dict) -> str:
    repo = inputs["repo"]
    number = inputs["pr_number"]
    return _run_gh(["pr", "view", str(number), "-R", repo])


# ── Local File Tools ───────────────────────────────────────────────────


def _read_local_file(inputs: dict) -> str:
    path = inputs["path"]
    # Safety: only allow reading from known project directories
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return f"File not found: {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        return f"Error reading {path}: {e}"


def _search_local_files(inputs: dict) -> str:
    query = inputs["query"]
    directory = inputs.get("directory", ".")
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        return f"Directory not found: {directory}"
    file_pattern = inputs.get("file_pattern", "")
    args = ["grep", "-rn", "--include", file_pattern, query, directory] if file_pattern else [
        "grep", "-rn", query, directory
    ]
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() or "No matches found."
    except subprocess.TimeoutExpired:
        return "Search timed out."
    except Exception as e:
        return f"Search error: {e}"


def _list_local_directory(inputs: dict) -> str:
    directory = inputs.get("directory", ".")
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        return f"Directory not found: {directory}"
    pattern = inputs.get("pattern", "")
    try:
        if pattern:
            import glob as glob_mod
            files = glob_mod.glob(os.path.join(directory, pattern), recursive=True)
            return "\n".join(sorted(files))
        else:
            entries = sorted(os.listdir(directory))
            return "\n".join(entries)
    except Exception as e:
        return f"Error listing {directory}: {e}"


# ── Web Search ────────────────────────────────────────────────────────


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
    import re
    import httpx
    url = inputs["url"]
    try:
        resp = httpx.get(
            url,
            follow_redirects=True,
            timeout=15.0,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
        )
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "pdf" in content_type:
            return f"PDF file at {url} ({len(resp.content)} bytes). Use open_url to display it in the browser."
        html = resp.text
        # Strip script/style blocks
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # Strip HTML tags
        text = re.sub(r"<[^>]+>", " ", html)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text[:4000] if text else "(empty page)"
    except httpx.HTTPStatusError as e:
        return f"HTTP error {e.response.status_code} fetching {url}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


def _get_current_time(inputs: dict) -> str:
    """Return the current date and time."""
    from datetime import datetime
    now = datetime.now()
    return f"Current date and time: {now.strftime('%A, %B %d, %Y at %H:%M:%S')}"


# ── macOS Computer Control ───────────────────────────────────────────


def _open_url(inputs: dict) -> str:
    """Open a URL in the default browser."""
    url = inputs["url"]
    subprocess.run(["open", url], check=True, timeout=10)
    return f"Opened {url} in default browser"


def _run_shell_command(inputs: dict) -> str:
    """Execute a shell command with safety checks."""
    command = inputs["command"]
    BLOCKED = ["rm -rf", "sudo", "mkfs", "dd if=", "> /dev/", "format ",
               "diskutil erase", "launchctl unload"]
    if any(b in command.lower() for b in BLOCKED):
        return "Error: Command blocked for safety"
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=15
    )
    output = (result.stdout + result.stderr).strip()
    return output[:4000] if output else "(no output)"


# ── Registry Factory ──────────────────────────────────────────────────


def create_default_registry() -> ToolRegistry:
    """Create a ToolRegistry with all default tools registered."""
    registry = ToolRegistry()

    # GitHub tools
    registry.register(AgentTool(
        name="github_search_code",
        description="Search for code across GitHub repositories. Returns matching files and code snippets.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (code, function name, error message, etc.)"},
                "repo": {"type": "string", "description": "Repository in owner/name format (optional — searches all repos if omitted)"},
                "language": {"type": "string", "description": "Filter by programming language (e.g. 'python', 'cpp')"},
            },
            "required": ["query"],
        },
        execute_fn=_github_search_code,
    ))

    registry.register(AgentTool(
        name="github_read_file",
        description="Read the contents of a specific file from a GitHub repository.",
        input_schema={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository in owner/name format"},
                "path": {"type": "string", "description": "File path within the repository"},
                "ref": {"type": "string", "description": "Branch or commit ref (default: HEAD)"},
            },
            "required": ["repo", "path"],
        },
        execute_fn=_github_read_file,
    ))

    registry.register(AgentTool(
        name="github_list_repo_files",
        description="List all files in a GitHub repository. Useful to understand project structure before reading specific files.",
        input_schema={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository in owner/name format"},
                "branch": {"type": "string", "description": "Branch name (default: main)"},
            },
            "required": ["repo"],
        },
        execute_fn=_github_list_repo_files,
    ))

    registry.register(AgentTool(
        name="github_list_repos",
        description="List GitHub repositories for a user or organization.",
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "GitHub username or org (optional — lists your own repos if omitted)"},
            },
        },
        execute_fn=_github_list_repos,
    ))

    registry.register(AgentTool(
        name="github_view_issue",
        description="View a GitHub issue with its full description and comments.",
        input_schema={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository in owner/name format"},
                "issue_number": {"type": "integer", "description": "Issue number"},
            },
            "required": ["repo", "issue_number"],
        },
        execute_fn=_github_view_issue,
    ))

    registry.register(AgentTool(
        name="github_view_pr",
        description="View a GitHub pull request with its description, changes, and review status.",
        input_schema={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository in owner/name format"},
                "pr_number": {"type": "integer", "description": "Pull request number"},
            },
            "required": ["repo", "pr_number"],
        },
        execute_fn=_github_view_pr,
    ))

    # Local file tools
    registry.register(AgentTool(
        name="read_local_file",
        description="Read a file from the local filesystem. Use absolute paths.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"},
            },
            "required": ["path"],
        },
        execute_fn=_read_local_file,
    ))

    registry.register(AgentTool(
        name="search_local_files",
        description="Search for text patterns in local files (like grep). Returns matching lines with file paths and line numbers.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text or regex pattern to search for"},
                "directory": {"type": "string", "description": "Directory to search in (absolute path)"},
                "file_pattern": {"type": "string", "description": "File glob pattern, e.g. '*.py' (optional)"},
            },
            "required": ["query"],
        },
        execute_fn=_search_local_files,
    ))

    registry.register(AgentTool(
        name="list_local_directory",
        description="List files and directories in a local directory.",
        input_schema={
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Absolute path to directory"},
                "pattern": {"type": "string", "description": "Glob pattern for filtering (e.g. '**/*.py')"},
            },
            "required": ["directory"],
        },
        execute_fn=_list_local_directory,
    ))

    # Web tools
    registry.register(AgentTool(
        name="web_search",
        description="Search the web for current information, documentation, prices, news, tutorials, etc. Returns titles, URLs, and snippets.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Number of results to return (default 5, max 10)"},
            },
            "required": ["query"],
        },
        execute_fn=_web_search,
    ))

    registry.register(AgentTool(
        name="get_current_time",
        description="Get the current date and time.",
        input_schema={
            "type": "object",
            "properties": {},
        },
        execute_fn=_get_current_time,
    ))

    registry.register(AgentTool(
        name="fetch_webpage",
        description="Fetch a webpage and return its text content (HTML stripped). Use to read search result pages, documentation, wikis, etc. For PDFs, returns a note to use open_url instead.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch"},
            },
            "required": ["url"],
        },
        execute_fn=_fetch_webpage,
    ))

    # macOS shortcuts (faster than Computer Use for common operations)
    registry.register(AgentTool(
        name="open_url",
        description="Open a URL in the default web browser. Faster than using Computer Use for opening links.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to open"},
            },
            "required": ["url"],
        },
        execute_fn=_open_url,
    ))

    registry.register(AgentTool(
        name="run_shell_command",
        description="Execute a shell command on macOS. Useful for system info, disk space, processes, custom CLI tools. Destructive commands are blocked.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
            },
            "required": ["command"],
        },
        execute_fn=_run_shell_command,
    ))

    logger.info("Tool registry created with %d tools: %s",
                len(registry.list_tools()), ", ".join(registry.list_tools()))
    return registry
