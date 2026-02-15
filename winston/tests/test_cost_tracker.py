"""Tests for CostTracker — daily API cost tracking and budget enforcement."""

from datetime import date


def test_record_and_get_daily_cost(cost_tracker):
    """Record tokens and verify cost calculation."""
    # FAST_INPUT_COST_PER_M = 1.00, so 1M input tokens = $1.00
    cost_tracker.record("fast", 1_000_000, 0)
    cost = cost_tracker.get_daily_cost()
    assert abs(cost - 1.00) < 0.001

    # FAST_OUTPUT_COST_PER_M = 5.00, so 1M output tokens = $5.00
    cost_tracker.record("fast", 0, 1_000_000)
    cost = cost_tracker.get_daily_cost()
    assert abs(cost - 6.00) < 0.001


def test_budget_exceeded(cost_tracker):
    """check_budget() returns False when daily cost exceeds MAX_DAILY_COST_USD ($2.00)."""
    assert cost_tracker.check_budget() is True

    # 2M fast input tokens → $2.00, which equals the budget
    # Then add a tiny bit more to push over (< comparison in check_budget)
    cost_tracker.record("fast", 2_000_001, 0)
    assert cost_tracker.check_budget() is False


def test_daily_report_format(cost_tracker):
    """get_daily_report() returns a human-readable summary with expected fields."""
    cost_tracker.record("fast", 500_000, 100_000)
    cost_tracker.record("smart", 50_000, 10_000)

    report = cost_tracker.get_daily_report()

    assert "Daily Cost Report" in report
    assert date.today().isoformat() in report
    assert "Haiku tokens:" in report
    assert "Sonnet tokens:" in report
    assert "$" in report
