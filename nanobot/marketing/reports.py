"""Report generation for market intelligence."""

from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader


class ReportGenerator:
    """Generates intelligence reports from templates."""

    def __init__(
        self,
        template_dir: str | Path | None = None,
    ):
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        self._template_dir = Path(template_dir)
        self._env = Environment(
            loader=FileSystemLoader(str(self._template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def generate_daily_brief(
        self,
        signals: list[dict[str, Any]],
        recommendations: list[dict[str, Any]],
        date: str | None = None,
    ) -> str:
        """Generate a daily intelligence brief."""
        template = self._env.get_template("daily_brief.md")
        return template.render(
            date=date or datetime.utcnow().strftime("%Y-%m-%d"),
            signals=signals,
            recommendations=recommendations,
            generated_at=datetime.utcnow().isoformat(),
        )

    def generate_weekly_report(
        self,
        signals: list[dict[str, Any]],
        top_leads: list[dict[str, Any]],
        recommendation_stats: dict[str, int],
        week_start: str = "",
        week_end: str = "",
        sent_count: int = 0,
        converted_count: int = 0,
    ) -> str:
        """Generate a weekly intelligence report."""
        signal_types: dict[str, int] = {}
        for s in signals:
            st = s.get("signal_type", "unknown")
            signal_types[st] = signal_types.get(st, 0) + 1

        template = self._env.get_template("weekly_report.md")
        return template.render(
            week_start=week_start,
            week_end=week_end,
            signal_count=len(signals),
            recommendation_count=sum(recommendation_stats.values()),
            sent_count=sent_count,
            converted_count=converted_count,
            signal_types=signal_types,
            top_leads=top_leads,
            recommendation_stats=recommendation_stats,
            generated_at=datetime.utcnow().isoformat(),
        )

    def generate_monthly_summary(
        self,
        total_signals: int = 0,
        total_recommendations: int = 0,
        approved_count: int = 0,
        sent_count: int = 0,
        conversions: int = 0,
        signal_trends: dict[str, int] | None = None,
        hot_count: int = 0,
        warm_count: int = 0,
        cold_count: int = 0,
        top_services: dict[str, int] | None = None,
        month_name: str = "",
        year: int = 0,
    ) -> str:
        """Generate a monthly summary report."""
        template = self._env.get_template("monthly_summary.md")
        now = datetime.utcnow()
        return template.render(
            month_name=month_name or now.strftime("%B"),
            year=year or now.year,
            total_signals=total_signals,
            total_recommendations=total_recommendations,
            approved_count=approved_count,
            sent_count=sent_count,
            conversions=conversions,
            signal_trends=signal_trends or {},
            hot_count=hot_count,
            warm_count=warm_count,
            cold_count=cold_count,
            top_services=top_services or {},
            generated_at=datetime.utcnow().isoformat(),
        )

    def render_outreach(self, template_name: str, variables: dict[str, Any]) -> str:
        """Render an outreach email template."""
        template = self._env.get_template(template_name)
        return template.render(**variables)

    def list_templates(self) -> list[str]:
        """List available template names."""
        return [
            p.name for p in self._template_dir.iterdir()
            if p.suffix == ".md" and p.is_file()
        ]
