"""Health check tool for nanobot error metrics and system status."""

from datetime import datetime
from pathlib import Path
from typing import Any

from nanobot.agent.errors import ErrorLogger, get_error_logger
from nanobot.agent.tools.base import Tool


class HealthCheckTool(Tool):
    """Health check tool providing error metrics, recovery rates, and status."""

    def __init__(self, workspace: Path):
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "health_check"

    @property
    def description(self) -> str:
        return "Check nanobot health status including error metrics and recovery rates"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "description": ("What to check: 'all', 'errors', 'recent', or 'summary'"),
                    "enum": ["all", "errors", "recent", "summary"],
                    "default": "summary",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 100,
                },
                "minutes": {
                    "type": "integer",
                    "description": "Time window in minutes for recent errors",
                    "default": 30,
                    "minimum": 1,
                    "maximum": 1440,
                },
            },
        }

    async def execute(self, **kwargs: Any) -> str:
        scope = kwargs.get("scope", "summary")
        limit = kwargs.get("limit", 10)
        minutes = kwargs.get("minutes", 30)

        error_logger = get_error_logger()
        if not error_logger:
            return "Error logger not initialized."

        if scope == "summary":
            return self._format_summary(error_logger.get_metrics())
        elif scope == "errors":
            return self._format_top_errors(error_logger.get_top_errors(limit), limit)
        elif scope == "recent":
            return self._format_recent_errors(
                error_logger.get_recent_errors(minutes), minutes, limit
            )
        elif scope == "all":
            return self._format_full_report(error_logger, limit, minutes)

        return "Unknown scope."

    def _format_summary(self, metrics: dict[str, Any]) -> str:
        total_errors = metrics.get("total_errors", 0)
        errors_last_hour = metrics.get("errors_last_hour", 0)

        if errors_last_hour == 0:
            status = "[HEALTHY] No errors in last hour"
        elif errors_last_hour < 5:
            status = "[GOOD] Less than 5 errors in last hour"
        elif errors_last_hour < 20:
            status = "[WARNING] 5-20 errors in last hour"
        else:
            status = "[CRITICAL] 20+ errors in last hour"

        recovery_rates = metrics.get("recovery_rates", {})
        worst_recovery = (
            min(recovery_rates, key=recovery_rates.get, default=None) if recovery_rates else None
        )
        worst_value = recovery_rates.get(worst_recovery, 1.0) if worst_recovery else 1.0

        report = [
            "# nanobot Health Summary",
            "",
            f"**Status:** {status}",
            "",
            f"**Total Errors:** {total_errors}",
            f"**Errors Last Hour:** {errors_last_hour}",
            f"**Error Rate:** {errors_last_hour} per hour",
            "",
        ]

        if worst_recovery and worst_value < 0.5:
            report.extend(
                [
                    "**Low Recovery Rate**",
                    f"Category '{worst_recovery}' only recovers "
                    f"{worst_value * 100:.0f}% of the time.",
                    "",
                ]
            )

        if errors_last_hour > 0:
            report.append("## Error Categories (Last Hour)")
            by_cat = metrics.get("errors_by_category", {})
            for category, count in by_cat.items():
                if count > 0:
                    report.append(f"- {category}: {count} errors")

        return "\n".join(report)

    def _format_top_errors(self, top_errors: list[dict[str, Any]], limit: int) -> str:
        report = ["# Top Error Categories", ""]

        if not top_errors:
            report.append("No errors recorded.")
            return "\n".join(report)

        report.append(f"Showing top {len(top_errors)} error categories:")
        report.append("")

        for i, err in enumerate(top_errors, 1):
            category = err.get("category", "unknown")
            count = err.get("count", 0)
            recovery = err.get("recovery_rate", 0.0) * 100

            if recovery > 80:
                tag = "[OK]"
            elif recovery > 50:
                tag = "[WARN]"
            else:
                tag = "[FAIL]"

            report.append(f"{i}. **{category}** ({count} errors, {recovery:.0f}% recovery) {tag}")

        report.extend(
            [
                "",
                "Recovery rate: how often the system auto-recovers.",
                "- [OK] >80%: Good automatic recovery",
                "- [WARN] 50-80%: Partial recovery",
                "- [FAIL] <50%: Needs investigation",
            ]
        )

        return "\n".join(report)

    def _format_recent_errors(
        self,
        recent_errors: list[dict[str, Any]],
        minutes: int,
        limit: int,
    ) -> str:
        report = [f"# Recent Errors (Last {minutes} minutes)", ""]

        if not recent_errors:
            report.append("No errors recorded in this time window.")
            return "\n".join(report)

        shown = min(len(recent_errors), limit)
        report.append(f"Showing last {shown} errors:")
        report.append("")

        for i, err in enumerate(recent_errors[:limit], 1):
            timestamp = err.get("timestamp", 0)
            ts = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S") if timestamp else "unknown"

            category = err.get("category", "unknown")
            message = err.get("error_message", "")[:100]
            tool = err.get("tool_name")
            severity = err.get("severity", "info")

            if severity == "critical":
                tag = "[CRIT]"
            elif severity == "error":
                tag = "[ERR]"
            else:
                tag = "[WARN]"

            tool_info = f" ({tool})" if tool else ""

            report.append(f"{i}. {tag} [{ts}] {category}{tool_info}")
            report.append(f"   {message}")

        return "\n".join(report)

    def _format_full_report(self, error_logger: ErrorLogger, limit: int, minutes: int) -> str:
        sections = [
            self._format_summary(error_logger.get_metrics()),
            "",
            self._format_top_errors(error_logger.get_top_errors(limit), limit),
            "",
            self._format_recent_errors(error_logger.get_recent_errors(minutes), minutes, limit),
        ]
        return "\n".join(sections)


def format_health_summary(metrics: dict[str, Any]) -> str:
    """Format health metrics into a one-line status string."""
    total_errors = metrics.get("total_errors", 0)
    errors_last_hour = metrics.get("errors_last_hour", 0)

    if errors_last_hour == 0:
        status = "HEALTHY"
    elif errors_last_hour < 5:
        status = "GOOD"
    elif errors_last_hour < 20:
        status = "WARNING"
    else:
        status = "CRITICAL"

    return f"{status} | {errors_last_hour} errors/hour | {total_errors} total"
