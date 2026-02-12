"""Structured error logging and categorization for nanobot self-evaluation."""

import json
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger


class ErrorCategory(Enum):
    """Categories of errors for tracking and analysis."""

    # Tool-related errors
    TOOL_EXECUTION = "tool_execution"
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_VALIDATION = "tool_validation"
    TOOL_NOT_FOUND = "tool_not_found"

    # LLM-related errors
    LLM_TIMEOUT = "llm_timeout"
    LLM_RATE_LIMIT = "llm_rate_limit"
    LLM_INVALID_RESPONSE = "llm_invalid_response"
    LLM_API_ERROR = "llm_api_error"

    # Memory-related errors
    MEMORY_INDEXING = "memory_indexing"
    MEMORY_RETRIEVAL = "memory_retrieval"
    MEMORY_CORRUPTION = "memory_corruption"
    EMBEDDING_ERROR = "embedding_error"

    # MCP-related errors
    MCP_SERVER_ERROR = "mcp_server_error"
    MCP_TOOL_CALL_FAILED = "mcp_tool_call_failed"
    MCP_TIMEOUT = "mcp_timeout"

    # System-related errors
    FILESYSTEM = "filesystem"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    CONFIGURATION = "configuration"

    # Daemon/cron related
    HEARTBEAT_ERROR = "heartbeat_error"
    CRON_JOB_ERROR = "cron_job_error"
    DAEMON_EXECUTION = "daemon_execution"

    # Other
    UNKNOWN = "unknown"


@dataclass
class ErrorRecord:
    """Structured error record for analysis."""

    timestamp: float
    category: ErrorCategory
    tool_name: str | None
    error_message: str
    error_type: str  # Exception class name
    context: dict[str, Any] | None
    severity: str  # "critical", "error", "warning", "info"
    recovered: bool  # Did the system recover automatically?


class ErrorLogger:
    """
    Centralized error logging with categorization and metrics.

    Logs to:
    - Loguru logger (traditional logging)
    - JSON error log (for analysis)
    - In-memory metrics (for health checks)
    """

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._error_log_path = workspace / "memory" / "errors.jsonl"
        self._error_log_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory metrics
        self._error_counts: dict[ErrorCategory, int] = {}
        self._error_rate_window: list[float] = []  # Errors in last hour
        self._recovery_counts: dict[ErrorCategory, int] = {}

    def log(
        self,
        category: ErrorCategory,
        error_message: str,
        error_type: str,
        tool_name: str | None = None,
        context: dict[str, Any] | None = None,
        severity: str = "error",
        recovered: bool = False,
    ) -> None:
        """
        Log an error with full context.

        Args:
            category: The error category for grouping
            error_message: Human-readable error message
            error_type: Exception class name
            tool_name: Name of tool if tool-related
            context: Additional context (arguments, state, etc.)
            severity: "critical", "error", "warning", "info"
            recovered: Did the system recover automatically?
        """
        timestamp = time.time()

        # Record to traditional logger
        if severity == "critical":
            logger.critical(f"[{category.value}] {error_message}")
        elif severity == "error":
            logger.error(f"[{category.value}] {error_message}")
        elif severity == "warning":
            logger.warning(f"[{category.value}] {error_message}")
        else:
            logger.info(f"[{category.value}] {error_message}")

        # Create structured record
        record = ErrorRecord(
            timestamp=timestamp,
            category=category,
            tool_name=tool_name,
            error_message=error_message,
            error_type=error_type,
            context=context,
            severity=severity,
            recovered=recovered,
        )

        # Write to JSONL log
        self._write_record(record)

        # Update in-memory metrics
        self._update_metrics(category, recovered)

    def log_exception(
        self,
        category: ErrorCategory,
        exception: Exception,
        tool_name: str | None = None,
        context: dict[str, Any] | None = None,
        recovered: bool = False,
    ) -> None:
        """
        Log an exception automatically extracting details.

        Args:
            category: The error category for grouping
            exception: The exception instance
            tool_name: Name of tool if tool-related
            context: Additional context (arguments, state, etc.)
            recovered: Did the system recover automatically?
        """
        self.log(
            category=category,
            error_message=str(exception),
            error_type=type(exception).__name__,
            tool_name=tool_name,
            context=context,
            severity="error",
            recovered=recovered,
        )

    def _write_record(self, record: ErrorRecord) -> None:
        """Write error record to JSONL log."""
        try:
            with open(self._error_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record)) + "\n")
        except Exception as e:
            # Fallback if we can't write to error log
            logger.error(f"Failed to write error record: {e}")

    def _update_metrics(self, category: ErrorCategory, recovered: bool) -> None:
        """Update in-memory error metrics."""
        # Count errors by category
        self._error_counts[category] = self._error_counts.get(category, 0) + 1

        # Track errors in time window (last hour)
        now = time.time()
        self._error_rate_window = [t for t in self._error_rate_window if now - t < 3600]
        self._error_rate_window.append(now)

        # Track recoveries
        if recovered:
            self._recovery_counts[category] = self._recovery_counts.get(category, 0) + 1

    def get_metrics(self) -> dict[str, Any]:
        """
        Get current error metrics for health checks.

        Returns:
            Dictionary with error statistics.
        """
        # Calculate error rate per hour
        errors_last_hour = len(self._error_rate_window)

        # Calculate recovery rate by category
        recovery_rates: dict[str, float] = {}
        for category, count in self._error_counts.items():
            recoveries = self._recovery_counts.get(category, 0)
            if count > 0:
                recovery_rates[category.value] = recoveries / count
            else:
                recovery_rates[category.value] = 0.0

        return {
            "total_errors": sum(self._error_counts.values()),
            "errors_by_category": {k.value: v for k, v in self._error_counts.items()},
            "errors_last_hour": errors_last_hour,
            "error_rate_per_hour": errors_last_hour,
            "recovery_rates": recovery_rates,
            "categories_with_errors": list(self._error_counts.keys()),
        }

    def get_top_errors(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get top error categories by count.

        Args:
            limit: Maximum number of categories to return

        Returns:
            List of categories sorted by error count.
        """
        sorted_categories = sorted(
            self._error_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:limit]

        return [
            {
                "category": cat.value,
                "count": count,
                "recovery_rate": self._recovery_counts.get(cat, 0) / count if count > 0 else 0,
            }
            for cat, count in sorted_categories
        ]

    def get_recent_errors(self, minutes: int = 30) -> list[dict[str, Any]]:
        """
        Get recent errors from JSONL log.

        Args:
            minutes: How many minutes back to look

        Returns:
            List of error records.
        """
        if not self._error_log_path.exists():
            return []

        cutoff = time.time() - (minutes * 60)
        records = []

        try:
            with open(self._error_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        if record.get("timestamp", 0) >= cutoff:
                            records.append(record)
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception as e:
            logger.warning(f"Failed to read error log: {e}")

        # Sort by timestamp descending
        records.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

        return records

    def reset_metrics(self) -> None:
        """Reset in-memory metrics (useful for testing)."""
        self._error_counts.clear()
        self._error_rate_window.clear()
        self._recovery_counts.clear()


# Global instance for singleton access
_global_error_logger: ErrorLogger | None = None


def get_error_logger() -> ErrorLogger | None:
    """Get the global error logger instance."""
    return _global_error_logger


def init_error_logger(workspace: Path) -> ErrorLogger:
    """Initialize the global error logger."""
    global _global_error_logger
    if _global_error_logger is None:
        _global_error_logger = ErrorLogger(workspace)
        logger.info(f"Error logger initialized: {workspace}")
    return _global_error_logger


# Convenience functions for logging


def log_error(
    category: ErrorCategory,
    message: str,
    tool_name: str | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    """Log an error (convenience function)."""
    logger = get_error_logger()
    if logger:
        logger.log(
            category=category,
            error_message=message,
            error_type="Error",
            tool_name=tool_name,
            context=context,
        )


def log_exception(
    category: ErrorCategory,
    exception: Exception,
    tool_name: str | None = None,
    context: dict[str, Any] | None = None,
    recovered: bool = False,
) -> None:
    """Log an exception (convenience function)."""
    logger = get_error_logger()
    if logger:
        logger.log_exception(
            category=category,
            exception=exception,
            tool_name=tool_name,
            context=context,
            recovered=recovered,
        )
