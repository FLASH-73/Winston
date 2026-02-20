"""Autonomous agent executor — runs Claude Opus with tools in an agentic loop.

Supports both regular tools (web_search, shell, file ops) and Anthropic Computer Use
(screenshot + mouse + keyboard). Uses the beta API when Computer Use is enabled.
"""

import base64
import logging
import time
from typing import TYPE_CHECKING, Callable, Optional

from brain.agent_tools import ToolRegistry

if TYPE_CHECKING:
    from brain.computer_use import ComputerController

logger = logging.getLogger("winston.agent")


class AgentExecutor:
    """Runs a Claude Opus agent with tools until it has an answer."""

    def __init__(
        self,
        client,
        cost_tracker,
        tool_registry: ToolRegistry,
        computer_controller: Optional["ComputerController"] = None,
    ):
        self._client = client  # anthropic.Anthropic instance
        self._cost_tracker = cost_tracker
        self._tools = tool_registry
        self._computer = computer_controller

    def run(
        self,
        task: str,
        system_prompt: Optional[str] = None,
        max_iterations: int = 20,
        max_tokens: int = 4096,
        on_progress: Optional[Callable[[str], None]] = None,
        on_screenshot: Optional[Callable[[bytes], None]] = None,
    ) -> Optional[str]:
        """Run the agentic loop until Claude provides a final answer.

        Args:
            task: The investigation task description.
            system_prompt: System prompt for the agent. Uses default if None.
            max_iterations: Maximum tool-use round trips.
            max_tokens: Max tokens per Claude response.
            on_progress: Optional callback for progress updates (tool names, etc.).
            on_screenshot: Optional callback receiving PNG bytes when Computer Use takes a screenshot.

        Returns:
            Final findings text, or None on failure.
        """
        from config import (
            AGENT_MAX_ITERATIONS,
            AGENT_MODEL,
            COMPUTER_USE_BETA,
            COMPUTER_USE_ENABLED,
        )

        if max_iterations == 20:  # default — use config value
            max_iterations = AGENT_MAX_ITERATIONS

        if system_prompt is None:
            from config import get_agent_prompt

            system_prompt = get_agent_prompt()

        messages = [{"role": "user", "content": task}]

        # Build tool definitions: regular tools + computer use tool
        tool_defs = self._tools.get_api_definitions()
        use_computer = COMPUTER_USE_ENABLED and self._computer is not None
        if use_computer:
            tool_defs.append(self._computer.get_tool_definition())

        logger.info(
            "[agent] Starting task with %d tools (computer_use=%s), max %d iterations",
            len(tool_defs),
            use_computer,
            max_iterations,
        )
        logger.info("[agent] Task: %.200s", task)

        t0 = time.time()
        consecutive_empty = 0

        for iteration in range(max_iterations):
            logger.info("[agent] Iteration %d/%d", iteration + 1, max_iterations)

            try:
                response = self._call_api(
                    use_computer,
                    AGENT_MODEL,
                    max_tokens,
                    system_prompt,
                    tool_defs,
                    messages,
                    COMPUTER_USE_BETA,
                )
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate_limit" in error_str:
                    # Rate limit — wait 30s and retry once (enough for per-minute limits)
                    logger.warning("[agent] Rate limited, waiting 30s before retry")
                    time.sleep(30)
                    try:
                        response = self._call_api(
                            use_computer,
                            AGENT_MODEL,
                            max_tokens,
                            system_prompt,
                            tool_defs,
                            messages,
                            COMPUTER_USE_BETA,
                        )
                    except Exception as retry_e:
                        logger.error("[agent] Retry also failed: %s", retry_e)
                        return None
                else:
                    logger.error("[agent] API call failed: %s", e)
                    return None

            self._track_usage(response)

            # Check if Claude is done (no more tool calls)
            if response.stop_reason != "tool_use":
                final_text = self._extract_text(response)
                elapsed = time.time() - t0
                logger.info(
                    "[agent] Complete in %.1fs (%d iterations): %.200s",
                    elapsed,
                    iteration + 1,
                    final_text or "<empty>",
                )
                return final_text

            # Extract and execute tool calls
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            text_parts = [b for b in response.content if hasattr(b, "text") and b.text]

            if text_parts:
                reasoning = " ".join(b.text for b in text_parts)
                logger.info("[agent] Reasoning: %.200s", reasoning)

            tool_results = []
            for tool_use in tool_uses:
                logger.info("[agent] Tool call: %s(%s)", tool_use.name, _short_json(tool_use.input))

                if on_progress:
                    try:
                        on_progress(f"Using {tool_use.name}...")
                    except Exception:
                        pass

                if tool_use.name == "computer" and self._computer:
                    # Computer Use tool — returns list of content blocks (text + image)
                    action = tool_use.input.get("action", "screenshot")
                    logger.info("[agent] Computer action: %s", action)
                    content_blocks = self._computer.execute(action, tool_use.input)
                    if on_screenshot:
                        for block in content_blocks:
                            if block.get("type") == "image":
                                try:
                                    img_bytes = base64.b64decode(block["source"]["data"])
                                    on_screenshot(img_bytes)
                                except Exception:
                                    pass
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": content_blocks,
                        }
                    )
                else:
                    # Regular tool — returns string
                    result = self._tools.execute(tool_use.name, tool_use.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": result,
                        }
                    )

            # Track consecutive empty results to detect stuck loops
            # Only check string results (computer use results contain images)
            empty_markers = ["no results found", "not found", "error:", "no files found", "no matches found"]
            string_results = [r for r in tool_results if isinstance(r["content"], str)]
            if string_results:
                empty_count = sum(1 for r in string_results if any(m in r["content"].lower() for m in empty_markers))
                if empty_count == len(string_results):
                    consecutive_empty += 1
                else:
                    consecutive_empty = 0
            # Computer use results always reset the counter (visual interaction = progress)
            else:
                consecutive_empty = 0

            # After 3 consecutive rounds of all-empty results, nudge Claude to wrap up
            if consecutive_empty >= 3:
                logger.info("[agent] %d consecutive empty rounds — injecting wrap-up hint", consecutive_empty)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_uses[-1].id,
                        "is_error": True,
                        "content": "SYSTEM: Multiple consecutive searches returned no results. "
                        "Wrap up with your best answer based on what you have, "
                        "or try a fundamentally different approach (different keywords, different language, different tool).",
                    }
                )
                consecutive_empty = 0

            # Append assistant response and tool results to messages
            messages.append({"role": "assistant", "content": _content_to_dicts(response.content)})
            messages.append({"role": "user", "content": tool_results})

        # Max iterations reached
        elapsed = time.time() - t0
        logger.warning("[agent] Max iterations (%d) reached in %.1fs", max_iterations, elapsed)

        last_text = self._extract_text(response) if response else None
        if last_text:
            return last_text + "\n\n(Note: investigation was cut short due to iteration limit)"
        return "Investigation reached the maximum number of steps without a conclusive finding."

    def _call_api(self, use_computer, model, max_tokens, system_prompt, tool_defs, messages, computer_use_beta):
        """Make agent API call (beta or standard)."""
        if use_computer:
            return self._client.beta.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                tools=tool_defs,
                messages=messages,
                betas=[computer_use_beta],
            )
        else:
            return self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                tools=tool_defs,
                messages=messages,
            )

    def _track_usage(self, response) -> None:
        """Track token usage for cost accounting."""
        try:
            usage = response.usage
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            self._cost_tracker.record("agent", input_tokens, output_tokens)
            logger.debug("[agent] Tokens: %d in / %d out", input_tokens, output_tokens)
        except Exception as e:
            logger.debug("[agent] Could not track usage: %s", e)

    @staticmethod
    def _extract_text(response) -> Optional[str]:
        """Extract text content from a Claude response."""
        texts = []
        for block in response.content:
            if hasattr(block, "text") and block.text:
                texts.append(block.text)
        return "\n".join(texts).strip() if texts else None


def _content_to_dicts(content_blocks) -> list[dict]:
    """Convert Anthropic ContentBlock objects to dicts for message history."""
    result = []
    for block in content_blocks:
        if block.type == "text":
            result.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            result.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
    return result


def _short_json(d: dict) -> str:
    """Short JSON representation for logging."""
    s = str(d)
    return s if len(s) <= 100 else s[:100] + "..."
