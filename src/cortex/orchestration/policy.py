"""Guardrails: decide which proposed actions run unattended.

Autonomy is a budget, not a switch. Every proposal gets one of three verdicts:

* auto   - safe enough to execute without a human
* review - queued for approval in the control plane
* deny   - never executed (e.g. destructive kinds an operator has ruled out)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from cortex.agents.base import ProposedAction, Risk
from cortex.config import PolicySettings

Verdict = Literal["auto", "review", "deny"]

_RISK_LEVEL = {"low": Risk.LOW, "medium": Risk.MEDIUM, "high": Risk.HIGH}


@dataclass(slots=True, frozen=True)
class PolicyDecision:
    action: ProposedAction
    verdict: Verdict
    reason: str


class ActionPolicy:
    def __init__(self, settings: PolicySettings) -> None:
        self.settings = settings
        self.max_auto_risk = _RISK_LEVEL[settings.max_auto_risk]

    def evaluate(
        self, actions: Iterable[ProposedAction], *, auto_actions_last_24h: int = 0
    ) -> list[PolicyDecision]:
        s = self.settings
        remaining = s.daily_auto_action_limit - auto_actions_last_24h
        decisions: list[PolicyDecision] = []
        for a in actions:
            change = a.params.get("change_pct")
            if a.kind in s.denied_kinds:
                decisions.append(PolicyDecision(a, "deny", f"'{a.kind}' is on the deny list"))
            elif a.risk > self.max_auto_risk:
                decisions.append(
                    PolicyDecision(
                        a,
                        "review",
                        f"risk {a.risk.name.lower()} is above auto limit {s.max_auto_risk}",
                    )
                )
            elif change is not None and abs(float(change)) > s.max_budget_change_pct:
                decisions.append(
                    PolicyDecision(
                        a, "review", f"{change:+g}% exceeds the {s.max_budget_change_pct:g}% cap"
                    )
                )
            elif remaining <= 0:
                decisions.append(PolicyDecision(a, "review", "daily auto-action limit reached"))
            else:
                remaining -= 1
                decisions.append(PolicyDecision(a, "auto", "within policy"))
        return decisions
