"""Deterministic lead scoring and recommendation engine."""

from dataclasses import dataclass, field
from typing import Any

# Signal type severity weights (higher = more consulting need)
SIGNAL_WEIGHTS: dict[str, float] = {
    "restructuring": 1.0,
    "leadership_change": 0.9,
    "ma": 0.8,
    "digital_transformation": 0.7,
    "expansion": 0.6,
    "cost_cutting": 0.5,
}

# K&P service areas mapped from signal types
SIGNAL_TO_SERVICE: dict[str, list[str]] = {
    "restructuring": ["turnaround", "change_management"],
    "leadership_change": ["leadership", "change_management"],
    "ma": ["strategy", "change_management", "process_optimization"],
    "digital_transformation": ["process_optimization", "change_management"],
    "expansion": ["strategy", "sales_management"],
    "cost_cutting": ["process_optimization", "turnaround"],
}

# K&P target industries and their weights
INDUSTRY_FIT: dict[str, float] = {
    "automotive": 1.0,
    "manufacturing": 0.95,
    "financial_services": 0.9,
    "healthcare": 0.85,
    "technology": 0.8,
    "energy": 0.8,
    "logistics": 0.75,
    "retail": 0.7,
    "telecommunications": 0.7,
    "pharma": 0.85,
}

# DACH region weight
REGION_FIT: dict[str, float] = {
    "germany": 1.0,
    "dach": 1.0,
    "austria": 0.9,
    "switzerland": 0.9,
    "europe": 0.6,
}


@dataclass
class ScoredLead:
    """A lead with computed scores and tier."""

    company_name: str
    signal_score: float = 0.0
    fit_score: float = 0.0
    engagement_score: float = 0.0
    total_score: float = 0.0
    tier: str = "cold"
    signal_count: int = 0
    top_signal_type: str = ""
    recommended_services: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Recommendation:
    """A generated outreach recommendation."""

    company_name: str
    consultant_name: str
    consultant_email: str
    service_area: str
    outreach_channel: str
    lead_score: float
    lead_tier: str
    signal_summary: str
    match_score: float = 0.0


class LeadScorer:
    """Deterministic, rule-based lead scoring.

    Score = signals (50%) + company_fit (30%) + engagement (20%)
    """

    SIGNAL_WEIGHT = 0.50
    FIT_WEIGHT = 0.30
    ENGAGEMENT_WEIGHT = 0.20

    def score_lead(
        self,
        company_name: str,
        signals: list[dict[str, Any]],
        company_info: dict[str, Any] | None = None,
        engagement_history: list[dict[str, Any]] | None = None,
    ) -> ScoredLead:
        """Score a lead based on signals, company fit, and engagement.

        Args:
            company_name: Name of the company.
            signals: List of signal dicts from intel store.
            company_info: Optional dict with industry, region, size.
            engagement_history: Optional list of past interactions.

        Returns:
            ScoredLead with computed scores and tier.
        """
        company_info = company_info or {}
        engagement_history = engagement_history or []

        signal_score = self._score_signals(signals)
        fit_score = self._score_company_fit(company_info)
        engagement_score = self._score_engagement(engagement_history)

        total = (
            signal_score * self.SIGNAL_WEIGHT
            + fit_score * self.FIT_WEIGHT
            + engagement_score * self.ENGAGEMENT_WEIGHT
        )

        # Clamp to [0, 1]
        total = max(0.0, min(1.0, total))

        tier = self._assign_tier(total)

        # Determine recommended services from top signal types
        services: list[str] = []
        for s in signals:
            st = s.get("signal_type", "")
            for svc in SIGNAL_TO_SERVICE.get(st, []):
                if svc not in services:
                    services.append(svc)

        # Top signal type by weight
        top_type = ""
        if signals:
            top_type = max(
                signals,
                key=lambda s: SIGNAL_WEIGHTS.get(s.get("signal_type", ""), 0),
            ).get("signal_type", "")

        return ScoredLead(
            company_name=company_name,
            signal_score=round(signal_score, 3),
            fit_score=round(fit_score, 3),
            engagement_score=round(engagement_score, 3),
            total_score=round(total, 3),
            tier=tier,
            signal_count=len(signals),
            top_signal_type=top_type,
            recommended_services=services,
            details={
                "signal_types": [s.get("signal_type") for s in signals],
                "industry": company_info.get("industry", ""),
                "region": company_info.get("region", ""),
            },
        )

    def _score_signals(self, signals: list[dict[str, Any]]) -> float:
        """Score based on signal count, type severity, and recency."""
        if not signals:
            return 0.0

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        weighted_sum = 0.0

        for signal in signals:
            signal_type = signal.get("signal_type", "")
            severity = SIGNAL_WEIGHTS.get(signal_type, 0.3)
            relevance = signal.get("relevance_score", 0.5)

            # Recency decay: full weight within 7 days, halves every 30 days
            recency = 1.0
            detected_at = signal.get("detected_at", "")
            if detected_at:
                try:
                    dt = datetime.fromisoformat(detected_at.replace("Z", "+00:00"))
                    days_old = (now - dt).days
                    if days_old > 7:
                        recency = 0.5 ** ((days_old - 7) / 30.0)
                except (ValueError, TypeError):
                    pass

            weighted_sum += severity * relevance * recency

        # Normalize: 1 strong signal = ~0.5, 3+ = approaching 1.0
        count_factor = min(1.0, len(signals) / 5.0)
        avg_weight = weighted_sum / len(signals)

        return min(1.0, avg_weight * 0.7 + count_factor * 0.3)

    def _score_company_fit(self, info: dict[str, Any]) -> float:
        """Score company fit based on industry, region, and size."""
        if not info:
            return 0.5  # Neutral if no info

        industry = info.get("industry", "").lower()
        region = info.get("region", "").lower()
        size = info.get("employees", 0)

        industry_score = INDUSTRY_FIT.get(industry, 0.4)
        region_score = REGION_FIT.get(region, 0.3)

        # Size scoring: K&P targets Mittelstand + large enterprises
        size_score = 0.5
        if size > 0:
            if size >= 500:
                size_score = 1.0
            elif size >= 100:
                size_score = 0.8
            elif size >= 50:
                size_score = 0.6
            else:
                size_score = 0.3

        return industry_score * 0.4 + region_score * 0.3 + size_score * 0.3

    def _score_engagement(self, history: list[dict[str, Any]]) -> float:
        """Score based on past engagement outcomes."""
        if not history:
            return 0.5  # Neutral if no history

        positive = 0
        negative = 0
        for interaction in history:
            outcome = interaction.get("outcome", "")
            if outcome in ("replied", "meeting", "proposal", "won"):
                positive += 1
            elif outcome in ("bounced", "unsubscribed", "rejected"):
                negative += 1

        total = positive + negative
        if total == 0:
            return 0.5

        return min(1.0, positive / total)

    @staticmethod
    def _assign_tier(score: float) -> str:
        """Assign lead tier based on score."""
        if score > 0.7:
            return "hot"
        if score > 0.4:
            return "warm"
        return "cold"


class RecommendationEngine:
    """Matches leads to K&P consultants for outreach."""

    def generate_recommendation(
        self,
        scored_lead: ScoredLead,
        consultants: list[dict[str, Any]],
    ) -> Recommendation | None:
        """Match a scored lead to the best consultant.

        Args:
            scored_lead: The scored lead to match.
            consultants: List of consultant dicts from intel store.

        Returns:
            Recommendation or None if no suitable match.
        """
        if not consultants:
            return None

        best_match: dict[str, Any] | None = None
        best_score = -1.0

        for consultant in consultants:
            if not consultant.get("active", True):
                continue

            score = self._match_score(scored_lead, consultant)
            if score > best_score:
                best_score = score
                best_match = consultant

        if not best_match:
            return None

        # Pick the top recommended service area
        service_area = ""
        if scored_lead.recommended_services:
            # Prefer services that match consultant specializations
            specs = best_match.get("specializations", [])
            for svc in scored_lead.recommended_services:
                if svc in specs:
                    service_area = svc
                    break
            if not service_area:
                service_area = scored_lead.recommended_services[0]

        # Build signal summary
        signal_summary = f"{scored_lead.signal_count} signal(s), top: {scored_lead.top_signal_type}"

        return Recommendation(
            company_name=scored_lead.company_name,
            consultant_name=best_match.get("name", ""),
            consultant_email=best_match.get("email", ""),
            service_area=service_area,
            outreach_channel="email",
            lead_score=scored_lead.total_score,
            lead_tier=scored_lead.tier,
            signal_summary=signal_summary,
            match_score=round(best_score, 3),
        )

    def _match_score(
        self,
        lead: ScoredLead,
        consultant: dict[str, Any],
    ) -> float:
        """Calculate consultant-lead match score."""
        specs = consultant.get("specializations", [])
        industries = consultant.get("industries", [])
        regions = consultant.get("regions", [])

        # Specialization overlap with recommended services
        spec_overlap = 0
        if lead.recommended_services and specs:
            common = set(lead.recommended_services) & set(specs)
            spec_overlap = len(common) / len(lead.recommended_services)

        # Industry match
        lead_industry = lead.details.get("industry", "")
        industry_match = 1.0 if lead_industry in industries else 0.0

        # Region match
        lead_region = lead.details.get("region", "")
        region_match = 1.0 if lead_region in regions else 0.0

        return spec_overlap * 0.5 + industry_match * 0.3 + region_match * 0.2
