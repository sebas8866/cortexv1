"""Ads optimizer: budget moves against CPA / ROAS targets, with memory of past moves.

What memory changes here:
* ``campaign:<id>.last_budget_change`` enforces a cooldown, so the agent does not
  scale the same campaign every cycle before the last change has had time to settle.
* ``campaign:<id>.status = paused`` stops it from re-proposing a pause.
* ``campaign:<id>.performance_tier`` is an observation; once corroborated across
  cycles it becomes a fact other agents (e.g. creative generation) can rely on.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from cortex.agents.base import Agent, AgentContext, AgentResult, ProposedAction, Risk
from cortex.agents.registry import register
from cortex.db import from_iso, to_iso
from cortex.memory import Observation


class AdsConfig(BaseModel):
    source: str = "examples/data/campaigns.csv"
    target_cpa: float = 40.0
    target_roas: float = 3.0
    winner_roas_multiple: float = 1.3
    loser_roas_multiple: float = 0.7
    min_conversions_to_scale: int = 5
    zero_conversion_spend_multiple: float = 2.0  # pause after spending N x target CPA with 0 conv
    scale_up_pct: float = 20.0
    scale_down_pct: float = 25.0
    budget_cooldown_hours: float = 48.0
    large_budget_threshold: float = 1000.0  # daily budgets above this are never LOW risk


@dataclass(slots=True, frozen=True)
class CampaignStats:
    """Trailing 7-day performance for one campaign."""

    campaign_id: str
    name: str
    channel: str
    daily_budget: float
    spend: float
    conversions: int
    revenue: float

    @property
    def roas(self) -> float:
        return self.revenue / self.spend if self.spend else 0.0

    @property
    def cpa(self) -> float | None:
        return self.spend / self.conversions if self.conversions else None


def load_campaigns(path: Path) -> list[CampaignStats]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [
            CampaignStats(
                campaign_id=row["campaign_id"],
                name=row["name"],
                channel=row["channel"],
                daily_budget=float(row["daily_budget"]),
                spend=float(row["spend"]),
                conversions=int(row["conversions"]),
                revenue=float(row["revenue"]),
            )
            for row in csv.DictReader(fh)
        ]


@register
class AdsOptimizerAgent(Agent):
    name = "ads_optimizer"
    description = "Reviews campaign performance against CPA/ROAS targets and proposes budget moves."

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.cfg = AdsConfig.model_validate(self.config)

    async def run(self, ctx: AgentContext, inputs: Mapping[str, Any]) -> AgentResult:
        cfg = self.cfg
        campaigns = load_campaigns(Path(str(inputs.get("source", cfg.source))))
        now = ctx.now()

        actions: list[ProposedAction] = []
        observations: list[Observation] = []
        notes: list[str] = []
        tiers = {"winner": 0, "underperformer": 0, "stable": 0, "no_signal": 0}

        for c in campaigns:
            subject = f"campaign:{c.campaign_id}"
            tier = self._tier(c)
            tiers[tier] += 1
            observations.append(Observation(subject, "performance_tier", tier, confidence=0.6))

            status = ctx.memory.get_fact(ctx.scope, subject, "status")
            if status and status.value == "paused":
                notes.append(f"{c.name}: paused earlier, skipped")
                continue

            if (
                c.conversions == 0
                and c.spend >= cfg.target_cpa * cfg.zero_conversion_spend_multiple
            ):
                actions.append(
                    ProposedAction(
                        agent=self.name,
                        kind="pause_campaign",
                        target=subject,
                        risk=Risk.MEDIUM,
                        rationale=(
                            f"${c.spend:,.0f} spent over 7 days with 0 conversions "
                            f"(limit {cfg.zero_conversion_spend_multiple:g}x target CPA "
                            f"${cfg.target_cpa:,.0f})"
                        ),
                        params={"campaign_id": c.campaign_id, "name": c.name},
                        effects=[Observation(subject, "status", "paused", 1.0)],
                    )
                )
                continue

            if tier not in ("winner", "underperformer"):
                continue

            held = self._cooldown_remaining(ctx, subject)
            if held is not None:
                notes.append(f"{c.name}: {tier}, budget change held ({held} of cooldown left)")
                continue

            pct = cfg.scale_up_pct if tier == "winner" else -cfg.scale_down_pct
            new_budget = round(c.daily_budget * (1 + pct / 100), 2)
            risk = Risk.MEDIUM if new_budget > cfg.large_budget_threshold else Risk.LOW
            actions.append(
                ProposedAction(
                    agent=self.name,
                    kind="increase_budget" if pct > 0 else "decrease_budget",
                    target=subject,
                    risk=risk,
                    rationale=(
                        f"ROAS {c.roas:.2f} vs target {cfg.target_roas:.2f} "
                        f"on {c.conversions} conversions"
                    ),
                    params={
                        "campaign_id": c.campaign_id,
                        "name": c.name,
                        "from": c.daily_budget,
                        "to": new_budget,
                        "change_pct": pct,
                    },
                    effects=[
                        Observation(subject, "daily_budget", f"{new_budget:.2f}", 1.0),
                        Observation(subject, "last_budget_change", to_iso(now), 1.0),
                    ],
                )
            )

        spend = sum(c.spend for c in campaigns)
        revenue = sum(c.revenue for c in campaigns)
        summary = (
            f"Reviewed {len(campaigns)} campaigns (7d spend ${spend:,.0f}, "
            f"blended ROAS {revenue / spend if spend else 0:.2f}): "
            f"{tiers['winner']} winners, {tiers['underperformer']} underperformers. "
            f"Proposed {len(actions)} actions; {len(notes)} held or skipped."
        )
        return AgentResult(
            agent=self.name,
            summary=summary,
            actions=actions,
            observations=observations,
            notes=notes,
            metrics={
                "campaigns": len(campaigns),
                "spend_7d": round(spend, 2),
                "revenue_7d": round(revenue, 2),
                "blended_roas": round(revenue / spend, 3) if spend else 0.0,
            },
        )

    def _tier(self, c: CampaignStats) -> str:
        cfg = self.cfg
        if c.spend < cfg.target_cpa:
            return "no_signal"
        if (
            c.roas >= cfg.target_roas * cfg.winner_roas_multiple
            and c.conversions >= cfg.min_conversions_to_scale
        ):
            return "winner"
        if c.roas < cfg.target_roas * cfg.loser_roas_multiple:
            return "underperformer"
        return "stable"

    def _cooldown_remaining(self, ctx: AgentContext, subject: str) -> str | None:
        last = ctx.memory.get_fact(ctx.scope, subject, "last_budget_change")
        if last is None:
            return None
        remaining = (
            from_iso(last.value) + timedelta(hours=self.cfg.budget_cooldown_hours) - ctx.now()
        )
        if remaining <= timedelta(0):
            return None
        return f"{remaining.total_seconds() / 3600:.0f}h"
