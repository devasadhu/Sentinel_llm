"""
dashboard/app.py
----------------
Streamlit dashboard for SentinelLLM.

HOW TO RUN:
    cd ~/sentinellm
    streamlit run dashboard/app.py

WHAT THIS DOES:
    - Reads all JSON reports from the reports/ directory
    - Displays attack results in an interactive table
    - Shows score distributions, success rates, risk breakdowns
    - Lets you drill into individual attack responses
    - Auto-refreshes when new reports are generated

WHY STREAMLIT?
    Streamlit turns Python scripts into web apps with zero HTML/CSS/JS.
    For a security tool demo, it's perfect — you can show live results
    in a browser while running attacks in the terminal.
    Real security tools like Streamlit: Pandas Profiling, ML monitoring
    dashboards, internal SOC tools.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

# Add project root to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Page config — must be first Streamlit call ────────────────────────────
st.set_page_config(
    page_title="SentinelLLM",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — makes it look professional ───────────────────────────────
st.markdown("""
<style>
    /* Dark header bar */
    .main-header {
        background: linear-gradient(90deg, #0d1117 0%, #161b22 100%);
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #00d4ff;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: #00d4ff;
        margin: 0;
        font-size: 1.8rem;
        font-family: monospace;
    }
    .main-header p {
        color: #8b949e;
        margin: 0.2rem 0 0 0;
        font-size: 0.85rem;
    }

    /* Metric cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }

    /* Status badges */
    .badge-success  { background:#1a4731; color:#3fb950; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600; }
    .badge-failure  { background:#4a1e1e; color:#f85149; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600; }
    .badge-partial  { background:#3d2b00; color:#e3b341; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600; }
    .badge-inconclusive { background:#1a1f36; color:#79c0ff; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600; }
    .badge-error    { background:#3d2b00; color:#ffa657; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600; }

    /* Risk badges */
    .risk-critical { color:#f85149; font-weight:700; }
    .risk-high     { color:#ffa657; font-weight:700; }
    .risk-medium   { color:#e3b341; font-weight:600; }
    .risk-low      { color:#3fb950; font-weight:500; }

    /* Response box */
    .response-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-family: monospace;
        font-size: 0.82rem;
        color: #c9d1d9;
        max-height: 200px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #30363d;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def load_all_reports() -> list[dict]:
    reports_dir = Path(__file__).parent.parent / "reports"
    reports = []
    if not reports_dir.exists():
        return []
    for f in sorted(reports_dir.glob("report_*.json"), reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                data["_filename"] = f.name
                data["_filepath"] = str(f)
                data["_type"] = "suite"
                reports.append(data)
        except Exception:
            pass
    return reports

@st.cache_data(ttl=10)
def load_benchmark_reports() -> list[dict]:
    reports_dir = Path(__file__).parent.parent / "reports"
    reports = []
    if not reports_dir.exists():
        return []
    for f in sorted(reports_dir.glob("benchmark_*.json"), reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                data["_filename"] = f.name
                reports.append(data)
        except Exception:
            pass
    return reports


def flatten_results(report: dict) -> pd.DataFrame:
    """Convert a report's results list into a flat DataFrame for display."""
    rows = []
    for r in report.get("results", []):
        rows.append({
            "ID":           r.get("attack_id", ""),
            "Name":         r.get("attack_name", ""),
            "Type":         r.get("attack_type", ""),
            "Category":     r.get("attack_category", ""),
            "Status":       r.get("status", "").upper(),
            "Score":        r.get("score", 0.0),
            "Risk":         r.get("risk_level", "LOW"),
            "Severity":     r.get("severity", ""),
            "MITRE":        r.get("mitre_tactic_id", ""),
            "OWASP":        r.get("owasp_id", ""),
            "Latency (ms)": r.get("latency_ms", 0.0),
            "Indicators":   ", ".join(r.get("indicators_found", [])),
            "Payload":      r.get("payload_text", "")[:120] + "..." if len(r.get("payload_text","")) > 120 else r.get("payload_text",""),
            "Response":     r.get("llm_response", ""),
            "Full Payload": r.get("payload_text", ""),
        })
    return pd.DataFrame(rows)


def status_badge(status: str) -> str:
    cls = f"badge-{status.lower()}"
    return f'<span class="{cls}">{status}</span>'


def risk_badge(risk: str) -> str:
    cls = f"risk-{risk.lower()}"
    return f'<span class="{cls}">{"⬛" if risk=="LOW" else "🟨" if risk=="MEDIUM" else "🟧" if risk=="HIGH" else "🟥"} {risk}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🛡️ SentinelLLM")
    st.markdown("*AI Security Testing Framework*")
    st.divider()

    reports = load_all_reports()

    if not reports:
        st.warning("No reports found.\n\nRun an attack first:\n```\npython -m cli.sentinel run --attack injection\n```")
        st.stop()

    # Report selector
    report_labels = []
    for r in reports:
        ts = r.get("timestamp", "")[:16].replace("T", " ")
        name = r.get("suite_name", "unknown").upper()
        report_labels.append(f"{name} — {ts}")

    selected_idx = st.selectbox(
        "Select Report",
        range(len(reports)),
        format_func=lambda i: report_labels[i],
    )
    report = reports[selected_idx]

    st.divider()

    # Quick stats in sidebar
    summary = report.get("summary", {})
    st.metric("Total Attacks",   summary.get("total", 0))
    st.metric("Succeeded",       summary.get("successful", 0))
    st.metric("Success Rate",    f"{summary.get('success_rate', 0)*100:.1f}%")
    st.metric("Avg Score",       f"{summary.get('average_score', 0):.3f}")

    st.divider()

    # Run new attack directly from dashboard
    st.markdown("**Run New Attack**")
    attack_choice = st.selectbox("Attack Type", ["injection", "jailbreak", "all"])
    if st.button("▶ Run Attack", type="primary", use_container_width=True):
        st.info(f"Run this in your terminal:\n\n`python -m cli.sentinel run --attack {attack_choice}`\n\nThen refresh this page.")

    if st.button("🔄 Refresh Reports", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ── Main content ──────────────────────────────────────────────────────────

# Header
st.markdown(f"""
<div class="main-header">
    <h1>🛡️ SentinelLLM Security Dashboard</h1>
    <p>AI Red Teaming & LLM Vulnerability Analysis Platform &nbsp;|&nbsp;
       Suite: <strong>{report.get('suite_name','').upper()}</strong> &nbsp;|&nbsp;
       Model: <strong>{report.get('model_name','')}</strong> &nbsp;|&nbsp;
       {report.get('timestamp','')[:16].replace('T',' ')} UTC
    </p>
</div>
""", unsafe_allow_html=True)


# ── Top metrics row ───────────────────────────────────────────────────────
summary = report.get("summary", {})
risk    = report.get("risk_summary", {})

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
col1.metric("Total",        summary.get("total", 0))
col2.metric("✅ Succeeded", summary.get("successful", 0))
col3.metric("❌ Failed",    summary.get("failed", 0))
col4.metric("⚠️ Partial",   summary.get("total",0) - summary.get("successful",0) - summary.get("failed",0) - summary.get("errors",0) - summary.get("inconclusive",0))
col5.metric("🔴 Critical",  risk.get("CRITICAL", 0))
col6.metric("🟠 High",      risk.get("HIGH", 0))
col7.metric("📊 Avg Score", f"{summary.get('average_score', 0):.3f}")

st.divider()


# ── Charts row ────────────────────────────────────────────────────────────
df = flatten_results(report)

if df.empty:
    st.warning("No results in this report.")
    st.stop()

chart_col1, chart_col2, chart_col3 = st.columns(3)

with chart_col1:
    st.markdown("**Attack Status Distribution**")
    status_counts = df["Status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    color_map = {
        "SUCCESS": "#3fb950", "FAILURE": "#f85149",
        "PARTIAL": "#e3b341", "INCONCLUSIVE": "#79c0ff", "ERROR": "#ffa657"
    }
    st.bar_chart(
        status_counts.set_index("Status"),
        color="#00d4ff",
        height=220,
    )

with chart_col2:
    st.markdown("**Score per Attack**")
    score_df = df[["ID", "Score"]].set_index("ID")
    st.bar_chart(score_df, color="#00d4ff", height=220)

with chart_col3:
    st.markdown("**Risk Level Breakdown**")
    risk_df = pd.DataFrame({
        "Risk Level": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        "Count": [risk.get("CRITICAL",0), risk.get("HIGH",0), risk.get("MEDIUM",0), risk.get("LOW",0)]
    }).set_index("Risk Level")
    st.bar_chart(risk_df, color="#ff6b6b", height=220)

st.divider()


# ── Filters ───────────────────────────────────────────────────────────────
st.markdown("### Attack Results")

filter_col1, filter_col2, filter_col3 = st.columns(3)
with filter_col1:
    status_filter = st.multiselect(
        "Filter by Status",
        options=df["Status"].unique().tolist(),
        default=df["Status"].unique().tolist(),
    )
with filter_col2:
    severity_filter = st.multiselect(
        "Filter by Severity",
        options=df["Severity"].unique().tolist(),
        default=df["Severity"].unique().tolist(),
    )
with filter_col3:
    score_min = st.slider("Minimum Score", 0.0, 1.0, 0.0, 0.05)

filtered_df = df[
    (df["Status"].isin(status_filter)) &
    (df["Severity"].isin(severity_filter)) &
    (df["Score"] >= score_min)
]


# ── Results table ─────────────────────────────────────────────────────────
display_cols = ["ID", "Name", "Status", "Score", "Risk", "Severity", "MITRE", "Latency (ms)", "Indicators"]
st.dataframe(
    filtered_df[display_cols],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Score": st.column_config.ProgressColumn(
            "Score", min_value=0, max_value=1, format="%.2f"
        ),
        "Latency (ms)": st.column_config.NumberColumn(
            "Latency (ms)", format="%.0f ms"
        ),
    }
)

st.divider()


# ── Attack detail drilldown ───────────────────────────────────────────────
st.markdown("### Attack Detail Inspector")
st.caption("Select an attack ID to inspect the full payload and LLM response")

attack_ids = filtered_df["ID"].tolist()
if attack_ids:
    selected_id = st.selectbox("Select Attack", attack_ids)
    row = filtered_df[filtered_df["ID"] == selected_id].iloc[0]

    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f"**{row['ID']} — {row['Name']}**")
        st.markdown(f"Status: `{row['Status']}` | Score: `{row['Score']:.3f}` | Risk: `{row['Risk']}`")
        st.markdown(f"MITRE: `{row['MITRE']}` | OWASP: `{row['OWASP']}` | Severity: `{row['Severity']}`")
        st.markdown(f"Latency: `{row['Latency (ms)']:.0f}ms`")
        if row["Indicators"]:
            st.markdown(f"**Indicators found:** `{row['Indicators']}`")
        st.markdown("**Payload sent to LLM:**")
        st.code(row["Full Payload"], language=None)

    with d2:
        st.markdown("**LLM Response:**")
        response = row["Response"]
        if response:
            st.text_area(
                label="response",
                value=response,
                height=300,
                label_visibility="collapsed",
            )
        else:
            st.info("No response recorded (error or empty)")

st.divider()


# ── MITRE ATLAS coverage ──────────────────────────────────────────────────
st.markdown("### MITRE ATLAS Coverage")
mitre_counts = df.groupby("MITRE")["ID"].count().reset_index()
mitre_counts.columns = ["Tactic ID", "Attack Count"]
mitre_names = {
    "AML.T0051": "LLM Prompt Injection",
    "AML.T0054": "Jailbreak ML Model",
    "AML.T0052": "Discover ML Model Ontology",
}
mitre_counts["Tactic Name"] = mitre_counts["Tactic ID"].map(mitre_names).fillna("Unknown")
mitre_counts["Succeeded"] = mitre_counts["Tactic ID"].map(
    df[df["Status"]=="SUCCESS"].groupby("MITRE")["ID"].count()
).fillna(0).astype(int)
st.dataframe(mitre_counts, use_container_width=True, hide_index=True)

st.divider()

# ── Benchmark tab ─────────────────────────────────────────────────────────
st.divider()
st.markdown("### Multi-Model Benchmark Comparison")
bench_reports = load_benchmark_reports()
if bench_reports:
    bench = bench_reports[0]
    import pandas as pd
    bench_rows = []
    for m in bench.get("models", []):
        bench_rows.append({
            "Model": m["model"],
            "Injection Rate": f"{m['injection']['success_rate']*100:.1f}%",
            "Jailbreak Rate": f"{m['jailbreak']['success_rate']*100:.1f}%",
            "Overall Vuln %": f"{m['overall_vulnerability']*100:.1f}%",
            "Successful Attacks": ", ".join(m["successful_attacks"]) or "none",
        })
    st.dataframe(pd.DataFrame(bench_rows), use_container_width=True, hide_index=True)
    st.caption(f"Benchmark run: {bench.get('timestamp','')[:16].replace('T',' ')} UTC")
else:
    st.info("No benchmark reports yet. Run: `python -m cli.sentinel benchmark`")

# ── Defense recommendations ────────────────────────────────────────────────
st.divider()
st.markdown("### Defense Recommendations")
successful_ids = df[df["Status"] == "SUCCESS"]["ID"].tolist()
if successful_ids:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from analysis.defense_advisor import get_recommendations
    recs = get_recommendations(successful_ids)
    for rec in recs:
        with st.expander(f"🔴 [{rec.attack_id}] {rec.attack_title} — {rec.category}"):
            st.markdown(f"**Remediation:** {rec.remediation}")
            st.code(rec.code_snippet, language="python")
else:
    st.success("No successful attacks in this report — no remediations needed.")

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown(
    "<p style='text-align:center; color:#8b949e; font-size:0.8rem;'>"
    "SentinelLLM — AI Security Testing Framework &nbsp;|&nbsp; "
    "OWASP LLM Top 10 &nbsp;|&nbsp; MITRE ATLAS aligned &nbsp;|&nbsp; "
    "Built with Python + Streamlit + Ollama"
    "</p>",
    unsafe_allow_html=True
)
