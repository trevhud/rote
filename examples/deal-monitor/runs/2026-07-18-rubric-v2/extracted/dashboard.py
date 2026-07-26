"""
HTML dashboard rendering and file I/O for deal-monitor.

render_dashboard: pure_function — fixed template, no external calls.
write_dashboard: external_call — filesystem write with error handling.
generate_summary: pure_function — fixed text template.

Compiled from: SKILL.md > Step 5 (Generate HTML Dashboard) and Step 6 (Output).
"""
from __future__ import annotations
import os
import html
from dataclasses import dataclass

DASHBOARD_OUTPUT_PATH: str = "./outputs/deal-monitor.html"

# Pipeline step enum values — kept here so render_dashboard can compute needs_attention
# without importing from the LLM signature module.
NEEDS_ATTENTION_STEPS: set[int] = {2, 3}  # Response Needed, Follow-Up Needed
STEP_NAMES: dict[int, str] = {
    1: "RFP Sent",
    2: "Response Needed",
    3: "Follow-Up Needed",
    4: "Pricing Returned",
    5: "Pricing Sent",
    6: "Declined",
}


@dataclass
class DashboardData:
    html_content: str
    new_opp_count: int
    quoting_count: int
    needs_attention_count: int
    closed_count: int


def render_dashboard(
    scored_opps: list[dict],         # from score_new_opportunity (unmatched opps + scores)
    classified_threads: list[dict],  # from classify_warehouse_thread (per-thread classification)
) -> DashboardData:
    """
    Render a self-contained three-tab HTML dashboard.

    Tabs:
    - New Opportunities: scored_opps (no matched email threads)
    - Quoting: classified_threads with pipeline_step 1-5
    - Closed: classified_threads with pipeline_step == 6 (Declined)

    Color coding: rows with pipeline_step in NEEDS_ATTENTION_STEPS get
    the .needs-attention CSS class.
    """
    quoting = [t for t in classified_threads if t.get("pipeline_step", 0) != 6]
    closed = [t for t in classified_threads if t.get("pipeline_step", 0) == 6]
    needs_attention_count = sum(
        1 for t in quoting if t.get("pipeline_step") in NEEDS_ATTENTION_STEPS
    )

    def _esc(v: object) -> str:
        return html.escape(str(v or ""))

    def _row_class(step: int) -> str:
        return ' class="needs-attention"' if step in NEEDS_ATTENTION_STEPS else ""

    # ── New Opportunities table ────────────────────────────────────────────────
    new_rows = ""
    for opp in scored_opps:
        risk_badge = ' <span class="risk-badge">SKU unknown</span>' if opp.get("sku_count_unknown") else ""
        new_rows += (
            f"<tr>"
            f"<td>{_esc(opp.get('account_name'))}{risk_badge}</td>"
            f"<td>{_esc(opp.get('submitter'))}</td>"
            f"<td>{_esc(opp.get('arr'))}</td>"
            f"<td>{_esc(opp.get('requested_location'))}</td>"
            f"<td>{_esc(opp.get('score'))}</td>"
            f"<td>{_esc(opp.get('tier'))}</td>"
            f"<td>{_esc(opp.get('key_factors'))}</td>"
            f"<td>{_esc(opp.get('risks'))}</td>"
            f"</tr>\n"
        )

    # ── Quoting table ──────────────────────────────────────────────────────────
    quoting_rows = ""
    for t in quoting:
        step = t.get("pipeline_step", 0)
        step_name = STEP_NAMES.get(step, str(step))
        thread_url = _esc(t.get("thread_url", ""))
        link = f'<a href="{thread_url}" target="_blank">{_esc(t.get("account_name", ""))}</a>' if thread_url else _esc(t.get("account_name", ""))
        quoting_rows += (
            f"<tr{_row_class(step)}>"
            f"<td>{link}</td>"
            f"<td>{_esc(t.get('warehouse'))}</td>"
            f"<td>{_esc(t.get('sent_date'))}</td>"
            f"<td>{step_name}</td>"
            f"<td>{_esc(t.get('classification_reason'))}</td>"
            f"</tr>\n"
        )

    # ── Closed table ───────────────────────────────────────────────────────────
    closed_rows = ""
    for t in closed:
        thread_url = _esc(t.get("thread_url", ""))
        link = f'<a href="{thread_url}" target="_blank">{_esc(t.get("account_name", ""))}</a>' if thread_url else _esc(t.get("account_name", ""))
        closed_rows += (
            f"<tr>"
            f"<td>{link}</td>"
            f"<td>{_esc(t.get('warehouse'))}</td>"
            f"<td>{_esc(t.get('sent_date'))}</td>"
            f"<td>{_esc(t.get('classification_reason'))}</td>"
            f"</tr>\n"
        )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Deal Monitor Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 16px; background: #f8f9fa; color: #212529; }}
  .tiles {{ display: flex; gap: 12px; margin-bottom: 20px; }}
  .tile {{ background: #fff; border-radius: 8px; padding: 16px 24px; box-shadow: 0 1px 3px rgba(0,0,0,.1); min-width: 120px; text-align: center; }}
  .tile .number {{ font-size: 2rem; font-weight: 700; }}
  .tile .label {{ font-size: .8rem; color: #6c757d; text-transform: uppercase; letter-spacing: .05em; }}
  .tile.alert .number {{ color: #dc3545; }}
  .tabs {{ display: flex; gap: 0; border-bottom: 2px solid #dee2e6; margin-bottom: 16px; }}
  .tab {{ padding: 10px 20px; cursor: pointer; border: none; background: none; font-size: 1rem; color: #6c757d; border-bottom: 2px solid transparent; margin-bottom: -2px; }}
  .tab.active {{ color: #0d6efd; border-bottom-color: #0d6efd; font-weight: 600; }}
  .panel {{ display: none; }}
  .panel.active {{ display: block; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
  th {{ background: #343a40; color: #fff; padding: 10px 12px; text-align: left; font-size: .85rem; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #e9ecef; font-size: .9rem; }}
  tr.needs-attention {{ background: #fff3cd; }}
  tr.needs-attention td {{ border-left: 3px solid #ffc107; }}
  tr:last-child td {{ border-bottom: none; }}
  a {{ color: #0d6efd; }}
  .risk-badge {{ background: #ffc107; color: #000; font-size: .7rem; padding: 1px 5px; border-radius: 4px; font-weight: 600; }}
</style>
</head>
<body>
<h1 style="margin-top:0">Deal Monitor</h1>
<div class="tiles">
  <div class="tile"><div class="number">{len(scored_opps)}</div><div class="label">New Opportunities</div></div>
  <div class="tile {'alert' if needs_attention_count > 0 else ''}"><div class="number">{len(quoting)}</div><div class="label">Active Quoting</div></div>
  <div class="tile {'alert' if needs_attention_count > 0 else ''}"><div class="number">{needs_attention_count}</div><div class="label">Needs Attention</div></div>
  <div class="tile"><div class="number">{len(closed)}</div><div class="label">Closed</div></div>
</div>
<div class="tabs">
  <button class="tab active" onclick="switchTab('new')">New Opportunities ({len(scored_opps)})</button>
  <button class="tab" onclick="switchTab('quoting')">Quoting ({len(quoting)})</button>
  <button class="tab" onclick="switchTab('closed')">Closed ({len(closed)})</button>
</div>
<div id="new" class="panel active">
  <table>
    <thead><tr><th>Account</th><th>Submitter</th><th>ARR</th><th>Location</th><th>Score</th><th>Tier</th><th>Key Factors</th><th>Risks</th></tr></thead>
    <tbody>{new_rows}</tbody>
  </table>
</div>
<div id="quoting" class="panel">
  <table>
    <thead><tr><th>Account</th><th>Warehouse</th><th>Sent Date</th><th>Pipeline Step</th><th>Notes</th></tr></thead>
    <tbody>{quoting_rows}</tbody>
  </table>
</div>
<div id="closed" class="panel">
  <table>
    <thead><tr><th>Account</th><th>Warehouse</th><th>Sent Date</th><th>Reason</th></tr></thead>
    <tbody>{closed_rows}</tbody>
  </table>
</div>
<script>
function switchTab(name) {{
  document.querySelectorAll('.tab').forEach((t, i) => t.classList.toggle('active', ['new','quoting','closed'][i] === name));
  document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', p.id === name));
}}
</script>
</body>
</html>"""

    return DashboardData(
        html_content=html_content,
        new_opp_count=len(scored_opps),
        quoting_count=len(quoting),
        needs_attention_count=needs_attention_count,
        closed_count=len(closed),
    )


def write_dashboard(dashboard: DashboardData, output_path: str = DASHBOARD_OUTPUT_PATH) -> str:
    """
    Write the rendered HTML to disk.

    Creates the output directory if it does not exist.
    Returns the absolute path of the written file.

    Compiled from: SKILL.md > Step 5 ("Save the HTML file to ./outputs/deal-monitor.html")
    """
    abs_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as fh:
        fh.write(dashboard.html_content)
    return abs_path


def generate_summary(dashboard: DashboardData) -> str:
    """
    Emit the brief text summary described in SKILL.md Step 6.

    Format:
      New opps: N | Quoting: M (K need attention) | Closed: P
    """
    attention_note = (
        f" ({dashboard.needs_attention_count} need attention)"
        if dashboard.needs_attention_count
        else ""
    )
    return (
        f"New opps: {dashboard.new_opp_count} | "
        f"Quoting: {dashboard.quoting_count}{attention_note} | "
        f"Closed: {dashboard.closed_count}"
    )
