"""Engagement Intelligence demo UI (Streamlit), a client of the HTTP service.

Run (with the service running):
  EI_API_URL=http://localhost:8000 EI_UI_AS_OF=2026-07-01 uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
from datetime import date, datetime, time, timezone

import httpx
import plotly.graph_objects as go
import streamlit as st

from engagement_intelligence.client import EngagementClient

API_URL = os.environ.get("EI_API_URL", "http://localhost:8000")
DEFAULT_AS_OF = date.fromisoformat(os.environ.get("EI_UI_AS_OF", date.today().isoformat()))
ACCENT = "#1a5c3a"
MUTED = "#b8c2bc"
BAND_COLORS = {"low": "#8a9a91", "medium": "#c8963e", "high": "#a8442f"}

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600&family=Outfit:wght@300;400&family=DM+Mono&display=swap');
h1, h2, h3, [data-testid="stHeading"] h1, [data-testid="stHeading"] h2, [data-testid="stHeading"] h3 {
  font-family: 'Cormorant Garamond', serif !important; color: #1a5c3a !important; font-weight: 600 !important;
}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stCaptionContainer"], [data-testid="stMetricLabel"] p, [data-testid="stWidgetLabel"] p,
button[data-baseweb="tab"] p {
  font-family: 'Outfit', sans-serif !important; font-weight: 300;
}
[data-testid="stMetricValue"] { font-family: 'DM Mono', monospace !important; }
.band { padding: 0.1rem 0.6rem; border-radius: 0.8rem; color: #ffffff; font-family: 'DM Mono', monospace; font-size: 0.8rem; }
</style>
"""


@st.cache_resource
def api() -> EngagementClient:
    return EngagementClient.connect(API_URL)


@st.cache_data(ttl=300)
def model_info() -> dict:
    return api().model_info().model_dump(mode="json")


@st.cache_data(ttl=300, show_spinner="Scoring accounts...")
def cohort(as_of_iso: str) -> dict:
    return api().assess(as_of=datetime.fromisoformat(as_of_iso), persist=False).model_dump(mode="json")


@st.cache_data(ttl=300)
def history(account_id: str) -> list[dict]:
    return [a.model_dump(mode="json") for a in api().risk_history(account_id)]


@st.cache_data(ttl=300)
def recent_events(account_id: str, as_of_iso: str, limit: int) -> list[dict]:
    as_of = datetime.fromisoformat(as_of_iso)
    return [e.model_dump(mode="json") for e in api().events(account_id, as_of=as_of, limit=limit)]


def styled(fig: go.Figure, title: str | None = None, height: int = 320) -> go.Figure:
    fig.update_layout(
        title=None if title is None else {"text": title, "font": {"family": "Cormorant Garamond, serif",
                                                                   "size": 20, "color": ACCENT}},
        font={"family": "Outfit, sans-serif", "color": "#1f2a24"},
        height=height, margin={"l": 10, "r": 10, "t": 40 if title else 10, "b": 10},
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
    )
    return fig


def band_badge(band: str) -> str:
    return f'<span class="band" style="background:{BAND_COLORS[band]}">{band.upper()}</span>'


def change_text(change: dict) -> str:
    if change["transition"] == "new":
        return "First assessment for this account."
    return (f"{change['transition'].capitalize()} (previous band: {change['previous_band']}, "
            f"assessed {change['previous_assessed_at'][:10]}).")


def render_cohort(data: dict, scored: list[dict], info: dict) -> None:
    summary, op = data["summary"], info["operating_point"]
    cols = st.columns(4)
    cols[0].metric("Accounts scored", summary["scored"])
    cols[1].metric("HIGH share", f"{summary['band_shares']['high']:.1%}")
    cols[2].metric("MEDIUM share", f"{summary['band_shares']['medium']:.1%}")
    cols[3].metric("Review queue", summary["within_capacity"],
                   help=f"Top {op['review_capacity']:.0%} of scored accounts by risk in this batch")
    left, right = st.columns([1, 2])
    with left:
        shares = summary["band_shares"]
        fig = go.Figure(go.Bar(x=list(shares), y=list(shares.values()),
                               marker_color=[BAND_COLORS[b] for b in shares]))
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(styled(fig, "Band distribution"), use_container_width=True)
    with right:
        st.subheader("Review queue")
        queue = sorted((a for a in scored if a["within_review_capacity"]), key=lambda a: -a["risk_score"])
        st.dataframe([{
            "account": a["account_id"],
            "risk": round(a["risk_score"], 3),
            "band": a["risk_band"],
            "change": a["risk_change"]["transition"],
            "main factor raising risk": a["top_drivers"][0]["description"] if a["top_drivers"] else "-",
        } for a in queue], hide_index=True, use_container_width=True, height=320)


def render_trajectory(account_id: str, as_of: datetime, op: dict) -> None:
    past = [h for h in history(account_id) if datetime.fromisoformat(h["assessed_at"]) <= as_of]
    if not any(h["status"] == "scored" for h in past):
        st.caption("No stored assessment history for this account up to the selected date.")
        return
    x = [h["assessed_at"][:10] for h in past]
    scored = [h["status"] == "scored" for h in past]
    fig = go.Figure(go.Scatter(
        x=x, y=[h["risk_score"] if s else None for h, s in zip(past, scored)],
        mode="lines+markers", connectgaps=False, line={"color": ACCENT}, name="risk score",
        marker={"size": 8, "color": [BAND_COLORS[h["risk_band"]] if s else MUTED for h, s in zip(past, scored)]}))
    gaps = [d for d, s in zip(x, scored) if not s]
    if gaps:
        fig.add_trace(go.Scatter(x=gaps, y=[0] * len(gaps), mode="markers", name="not eligible",
                                 marker={"symbol": "line-ns-open", "size": 10, "color": MUTED}))
    for band, key in (("high", "high_threshold"), ("medium", "medium_threshold")):
        fig.add_hline(y=op[key], line_dash="dot", line_color=BAND_COLORS[band],
                      annotation_text=f"{band.upper()} threshold")
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    st.plotly_chart(styled(fig, "Risk trajectory (stored weekly assessments)", 340), use_container_width=True)
    if gaps:
        st.caption("Gaps are weeks when the account was not eligible: no engagement known in the "
                   "previous eligibility window, so no risk score is produced.")


def render_account(scored: list[dict], as_of: datetime, info: dict) -> None:
    ranked = sorted(scored, key=lambda a: -a["risk_score"])
    account_id = st.selectbox("Account (sorted by risk)", [a["account_id"] for a in ranked])
    current = next(a for a in ranked if a["account_id"] == account_id)
    change = current["risk_change"]

    left, right = st.columns([1, 2])
    with left:
        st.markdown(f"### <span style=\"font-family:'DM Mono',monospace\">{account_id}</span> "
                    f"{band_badge(current['risk_band'])}", unsafe_allow_html=True)
        delta = None if change["score_delta"] is None else f"{change['score_delta']:+.1%}"
        st.metric(f"Probability of no engagement in the next {current['horizon_days']} days",
                  f"{current['risk_score']:.1%}", delta=delta, delta_color="inverse")
        st.caption(change_text(change))
        st.markdown("**Factors raising risk** (relative to a typical account)")
        drivers = current["top_drivers"]
        if drivers:
            fig = go.Figure(go.Bar(
                x=[d["contribution"] for d in drivers][::-1],
                y=[f"{d['description']} ({d['value']:.2f})" for d in drivers][::-1],
                orientation="h", marker_color=ACCENT))
            fig.update_xaxes(title="log-odds contribution")
            st.plotly_chart(styled(fig, height=220), use_container_width=True)
        else:
            st.caption("No factor raises risk above a typical account.")
    with right:
        render_trajectory(account_id, as_of, info["operating_point"])

    st.subheader("Recent events known at the assessment date")
    st.dataframe([{
        "occurred": e["timestamp"][:16].replace("T", " "),
        "recorded": e["recorded_at"][:16].replace("T", " "),
        "event": e["event_type"],
        "actor": e["actor"],
    } for e in recent_events(account_id, as_of.isoformat(), 30)], hide_index=True, use_container_width=True)


def render_model(info: dict) -> None:
    target, op, model = info["target"], info["operating_point"], info["model"]
    st.markdown(f"**Model:** `{model['name']}` · calibration: `{info['calibration']['method']}` · "
                f"trained on assessments from {info['training']['start']} to {info['training']['end']}")
    st.markdown(f"**Target:** no `{target['engagement_role']}` event within {target['horizon_days']} days; "
                f"assessed weekly ({target['assessment_weekday']}) for accounts engaged in the previous "
                f"{target['eligibility_window_days']} days.")
    st.markdown(f"**Bands (demo values):** HIGH at or above {op['high_threshold']:.3f}, MEDIUM at or above "
                f"{op['medium_threshold']:.3f}; review queue = top {op['review_capacity']:.0%} of each batch.")
    used = info["policy_dependent_features_used"]
    st.caption("Policy-dependent features used: " + (", ".join(used) if used else "none"))
    st.dataframe(info["features"], hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Engagement Intelligence", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    try:
        info = model_info()
    except httpx.HTTPError as exc:
        st.error(f"The scoring service is not reachable at {API_URL} ({exc}). Start it, then reload.")
        st.stop()

    st.title("Engagement Intelligence")
    st.caption(f"Model `{info['model_version']}` · synthetic demo data · bands use demo thresholds")
    with st.sidebar:
        as_of_day = st.date_input("Assessment date (00:00 UTC)", DEFAULT_AS_OF)
        st.caption(f"Service: {API_URL}")
    as_of = datetime.combine(as_of_day, time(), tzinfo=timezone.utc)

    data = cohort(as_of.isoformat())
    scored = [a for a in data["assessments"] if a["status"] == "scored"]
    if not scored:
        st.warning("No accounts are eligible for assessment at this date.")
        st.stop()

    tab_cohort, tab_account, tab_model = st.tabs(["Cohort", "Account", "Model"])
    with tab_cohort:
        render_cohort(data, scored, info)
    with tab_account:
        render_account(scored, as_of, info)
    with tab_model:
        render_model(info)


main()