"""Operator dashboard (Streamlit). Talks only to the control-plane API.

pip install -e ".[dashboard]"
CORTEX_API_URL=http://localhost:8000 streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API = os.environ.get("CORTEX_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Cortex", layout="wide")


def get(path: str, **params: Any) -> Any:
    response = httpx.get(f"{API}{path}", params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def post(path: str, body: dict[str, Any]) -> Any:
    response = httpx.post(f"{API}{path}", json=body, timeout=120)
    response.raise_for_status()
    return response.json()


try:
    health = get("/health")
except httpx.HTTPError as exc:
    st.error(f"Control plane unreachable at {API}: {exc}")
    st.stop()

st.title("Cortex")
st.caption(
    f"v{health['version']} | llm: {health['llm']} | {'DRY RUN' if health['dry_run'] else 'LIVE'}"
)

runs_tab, approvals_tab, memory_tab, audit_tab = st.tabs(
    ["Runs", "Approvals", "Memory", "Audit log"]
)

with runs_tab:
    agents = get("/agents")
    col1, col2 = st.columns([2, 1])
    agent = col1.selectbox("Agent", list(agents), format_func=lambda a: f"{a} - {agents[a]}")
    scope = col2.text_input("Scope", value="personal")
    if st.button("Run cycle now", type="primary"):
        with st.spinner("running"):
            run = post("/runs", {"agent": agent, "scope": scope})
        st.markdown(run["summary"] or f"**Failed:** {run['error']}")
    st.subheader("Recent runs")
    for run in get("/runs", limit=20):
        with st.expander(f"{run['started_at'][:19]}  {run['agent']}  [{run['status']}]"):
            st.markdown(run["summary"] or run["error"] or "")
            st.dataframe(run["trace"], use_container_width=True)

with approvals_tab:
    pending = get("/approvals")
    if not pending:
        st.info("Nothing waiting for a human.")
    reviewer = st.text_input("Your name (recorded in the audit log)", value="operator")
    for a in pending:
        action = a["action"]
        with st.container(border=True):
            st.markdown(f"**{action['kind']}** on `{action['target']}`  (risk: {action['risk']})")
            st.write(action["rationale"])
            st.caption(f"Held because: {a['reason']}")
            c1, c2, _ = st.columns([1, 1, 6])
            if c1.button("Approve", key=f"ok-{a['id']}"):
                post(f"/approvals/{a['id']}", {"approve": True, "by": reviewer})
                st.rerun()
            if c2.button("Reject", key=f"no-{a['id']}"):
                post(f"/approvals/{a['id']}", {"approve": False, "by": reviewer})
                st.rerun()

with memory_tab:
    mscope = st.text_input("Memory scope", value="personal", key="mscope")
    st.json(get("/memory/stats", scope=mscope))
    query = st.text_input("Recall", placeholder="e.g. paused campaign budget")
    if query:
        st.dataframe(get("/memory/search", q=query, scope=mscope), use_container_width=True)
    subject = st.text_input("Facts about subject", placeholder="campaign:c_101")
    if subject:
        st.dataframe(get(f"/memory/facts/{subject}", scope=mscope), use_container_width=True)

with audit_tab:
    st.dataframe(get("/executions", limit=200), use_container_width=True)
