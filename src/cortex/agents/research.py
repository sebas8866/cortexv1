"""Research agent: answers a question grounded in what Cortex already remembers.

It proposes no actions. Its value is synthesis: recall the relevant episodes, facts
and procedures, then have the LLM write a cited brief. With the offline provider it
returns the retrieved context itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cortex.agents.base import Agent, AgentContext, AgentResult
from cortex.agents.registry import register
from cortex.llm import BRIEFING_MARKER

_SYSTEM = (
    "You are the research function of an autonomous operations system. Answer only from the "
    "provided memory excerpts. Cite excerpt ids in brackets. Say plainly when memory is "
    "insufficient."
)


@register
class ResearchAgent(Agent):
    name = "research"
    description = "Answers questions from long-term memory and writes a cited brief."

    async def run(self, ctx: AgentContext, inputs: Mapping[str, Any]) -> AgentResult:
        question = str(inputs.get("question", "")).strip()
        if not question:
            return AgentResult(agent=self.name, summary="No question supplied.")

        hits = ctx.memory.recall(question, ctx.scope, limit=int(inputs.get("limit", 8)))
        excerpts = "\n".join(
            f"[{h.id}] ({h.kind.value}, score {h.score:.2f}) {h.text}" for h in hits
        )
        fallback = f"Memory excerpts for: {question}\n{excerpts or '(no matching memories)'}"
        brief = await ctx.llm.complete(
            system=_SYSTEM,
            prompt=f"Question: {question}\n\nMemory:\n{excerpts}\n{BRIEFING_MARKER}\n{fallback}",
        )
        return AgentResult(
            agent=self.name,
            summary=brief,
            metrics={"memories_used": len(hits)},
        )
