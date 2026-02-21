import json
import logging
import os
from datetime import date
from threading import Lock

from config import (
    AGENT_INPUT_COST_PER_M,
    AGENT_OUTPUT_COST_PER_M,
    FAST_INPUT_COST_PER_M,
    FAST_OUTPUT_COST_PER_M,
    MAX_DAILY_COST_USD,
    SMART_INPUT_COST_PER_M,
    SMART_OUTPUT_COST_PER_M,
)

logger = logging.getLogger("winston.cost_tracker")

COST_FILE = "winston_costs.json"


class CostTracker:
    def __init__(self):
        self._lock = Lock()
        self._data = self._load_or_reset()
        self._warned_50 = False
        self._warned_75 = False
        self._warned_90 = False

    def _load_or_reset(self) -> dict:
        """Load today's data from JSON file, or reset if it's a new day."""
        today = date.today().isoformat()
        if os.path.exists(COST_FILE):
            try:
                with open(COST_FILE, "r") as f:
                    data = json.load(f)
                if data.get("date") == today:
                    return data
            except (json.JSONDecodeError, KeyError):
                pass
        return self._empty_data(today)

    @staticmethod
    def _empty_data(today: str) -> dict:
        return {
            "date": today,
            "fast_input_tokens": 0,
            "fast_output_tokens": 0,
            "smart_input_tokens": 0,
            "smart_output_tokens": 0,
            "agent_input_tokens": 0,
            "agent_output_tokens": 0,
            "total_calls": 0,
        }

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """Record token usage from an API call. model is 'fast', 'smart', or 'agent'."""
        with self._lock:
            # Reset if day has changed
            today = date.today().isoformat()
            if self._data["date"] != today:
                self._data = self._empty_data(today)
                self._warned_50 = False
                self._warned_75 = False
                self._warned_90 = False

            self._data[f"{model}_input_tokens"] += input_tokens
            self._data[f"{model}_output_tokens"] += output_tokens
            self._data["total_calls"] += 1
            self._save()
            self._check_warnings()

    def check_budget(self) -> bool:
        """Return True if daily budget has NOT been exceeded."""
        return self.get_daily_cost() < MAX_DAILY_COST_USD

    def get_budget_state(self) -> str:
        """Return tiered budget state based on daily cost ratio.

        Returns:
            'normal'    - <50% of daily budget used
            'cautious'  - 50-80% used
            'critical'  - 80-100% used (background API calls should stop)
            'exhausted' - >=100% used (only interactive calls allowed)
        """
        cost = self.get_daily_cost()
        if MAX_DAILY_COST_USD <= 0:
            return "normal"
        ratio = cost / MAX_DAILY_COST_USD
        if ratio >= 1.0:
            return "exhausted"
        if ratio >= 0.8:
            return "critical"
        if ratio >= 0.5:
            return "cautious"
        return "normal"

    def get_daily_cost(self) -> float:
        """Return total estimated cost for today in USD."""
        with self._lock:
            return self._get_daily_cost_unlocked()

    def _get_daily_cost_unlocked(self) -> float:
        """Compute daily cost. Caller must already hold self._lock."""
        d = self._data
        return (
            d["fast_input_tokens"] * FAST_INPUT_COST_PER_M / 1_000_000
            + d["fast_output_tokens"] * FAST_OUTPUT_COST_PER_M / 1_000_000
            + d["smart_input_tokens"] * SMART_INPUT_COST_PER_M / 1_000_000
            + d["smart_output_tokens"] * SMART_OUTPUT_COST_PER_M / 1_000_000
            + d.get("agent_input_tokens", 0) * AGENT_INPUT_COST_PER_M / 1_000_000
            + d.get("agent_output_tokens", 0) * AGENT_OUTPUT_COST_PER_M / 1_000_000
        )

    def get_daily_report(self) -> str:
        """Return a human-readable summary of today's API usage."""
        with self._lock:
            d = self._data
            cost = self._get_daily_cost_unlocked()
            return (
                f"=== Daily Cost Report ({d['date']}) ===\n"
                f"Total API calls: {d['total_calls']}\n"
                f"Haiku tokens:  {d['fast_input_tokens']:,} in / {d['fast_output_tokens']:,} out\n"
                f"Sonnet tokens: {d['smart_input_tokens']:,} in / {d['smart_output_tokens']:,} out\n"
                f"Opus tokens:   {d.get('agent_input_tokens', 0):,} in / {d.get('agent_output_tokens', 0):,} out\n"
                f"Estimated cost: ${cost:.4f} / ${MAX_DAILY_COST_USD:.2f} budget"
            )

    def _save(self) -> None:
        """Persist data to JSON file. Must be called with lock held."""
        try:
            with open(COST_FILE, "w") as f:
                json.dump(self._data, f, indent=2)
        except OSError as e:
            logger.error(f"Failed to save cost data: {e}")

    def _check_warnings(self) -> None:
        """Log warnings at 50%, 75%, 90% of daily budget. Must be called with lock held."""
        cost = self._get_daily_cost_unlocked()
        ratio = cost / MAX_DAILY_COST_USD if MAX_DAILY_COST_USD > 0 else 0

        if ratio >= 0.90 and not self._warned_90:
            logger.warning(f"90% of daily budget used (${cost:.4f} / ${MAX_DAILY_COST_USD:.2f})")
            self._warned_90 = True
        elif ratio >= 0.75 and not self._warned_75:
            logger.warning(f"75% of daily budget used (${cost:.4f} / ${MAX_DAILY_COST_USD:.2f})")
            self._warned_75 = True
        elif ratio >= 0.50 and not self._warned_50:
            logger.info(f"50% of daily budget used (${cost:.4f} / ${MAX_DAILY_COST_USD:.2f})")
            self._warned_50 = True
