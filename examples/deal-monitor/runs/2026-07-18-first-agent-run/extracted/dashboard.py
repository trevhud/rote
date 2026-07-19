# Crystallized from SKILL.md Step 5-6.
# Fixed 3-tab HTML template with summary tile row, color-coded rows.
# All data is structured at this point — no LLM needed.
from __future__ import annotations

import os
from typing import Any

DASHBOARD_OUTPUT_PATH = "./outputs/deal-monitor.html"  # SKILL.md:53

# Pipeline step numbers that require attention (highlighted in red/amber)
NEEDS_ATTENTION_STEPS = frozenset({2, 3})  # step 2=Response Needed, 3=Follow-Up Needed
STEP_LABELS = {
    1: "RFP Sent",
    2: "Response Needed",
    3: "Follow-Up Needed",
    4: "Pricing Returned",
    5: "Pricing Sent",
    6: "Declined",
}
STEP_COLORS = {
    1: "#f0f4ff",  # neutral blue
    2: "#fff3cd",  # amber — needs attention
    3: "#fff3cd",  # amber — needs attention
    4: "#d1ecf1",  # teal
    5: "#d4edda",  # green
    6: "#f8d7da",  # red
}


def _escape(text: str | None) -> str:
    if not text:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def generate_dashboard_html(
    new_opps: list[Any],   # list of ScoredDeal (deal + score + risk_flags)
    quoting: list[Any],    # list of ClassifiedDeal (deal + threads + step)
    sku_risk_flags: list[str] | None = None,
) -> str:
    """Render the three-tab HTML dashboard from structured data.

    Tabs: New Opportunities | Quoting | Closed
    Summary tiles: total new opps, active quoting (needs attention), closed count.
    Color-coded rows based on pipeline step (from STEP_COLORS).
    Self-contained — no external assets.
    """
    closed = [d for d in quoting if d.step == 6]
    active = [d for d in quoting if d.step != 6]
    needs_attention = [d for d in active if d.step in NEEDS_ATTENTION_STEPS]

    def new_opps_rows() -> str:
        rows = []
        for item in new_opps:
            d = item.deal
            risk_badge = ""
            if d.account_name in (sku_risk_flags or []):
                risk_badge = ' <span style="color:#856404;background:#fff3cd;padding:1px 6px;border-radius:3px;font-size:11px;">SKU risk</span>'
            rows.append(
                f"<tr>"
                f"<td>{_escape(d.account_name)}{risk_badge}</td>"
                f"<td>{_escape(d.submitter)}</td>"
                f"<td>{_escape(str(d.arr or ''))}</td>"
                f"<td>{_escape(d.requested_location)}</td>"
                f"<td>{_escape(str(d.sku_count or 'unknown'))}</td>"
                f"<td>{_escape(str(d.go_live_date or ''))}</td>"
                f"<td><b>{_escape(item.score)}</b></td>"
                f"<td>{_escape(item.rationale)}</td>"
                f"<td>{_escape(', '.join(item.risk_flags))}</td>"
                f"</tr>"
            )
        return "\n".join(rows) if rows else '<tr><td colspan="9" style="text-align:center;color:#666">No new opportunities</td></tr>'

    def quoting_rows(deals: list[Any]) -> str:
        rows = []
        for item in deals:
            d = item.deal
            step = item.step
            color = STEP_COLORS.get(step, "#fff")
            label = STEP_LABELS.get(step, str(step))
            for thread in item.threads:
                rows.append(
                    f'<tr style="background:{color}">'
                    f"<td>{_escape(d.account_name)}</td>"
                    f"<td>{_escape(thread.account_name)}</td>"
                    f"<td><b>{step} — {_escape(label)}</b></td>"
                    f"<td>{_escape(str(thread.sent_date or ''))}</td>"
                    f'<td><a href="{_escape(thread.gmail_thread_url)}" target="_blank">Open</a></td>'
                    f"</tr>"
                )
        return "\n".join(rows) if rows else '<tr><td colspan="5" style="text-align:center;color:#666">No active quoting</td></tr>'

    def closed_rows() -> str:
        rows = []
        for item in closed:
            d = item.deal
            for thread in item.threads:
                rows.append(
                    f'<tr style="background:#f8d7da">'
                    f"<td>{_escape(d.account_name)}</td>"
                    f'<td><a href="{_escape(thread.gmail_thread_url)}" target="_blank">Open</a></td>'
                    f"</tr>"
                )
        return "\n".join(rows) if rows else '<tr><td colspan="2" style="text-align:center;color:#666">No closed deals</td></tr>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Deal Monitor Dashboard</title>
<style>
  body {{font-family:system-ui,sans-serif;margin:0;padding:20px;background:#f8f9fa}}
  h1 {{font-size:20px;margin-bottom:16px}}
  .tiles {{display:flex;gap:12px;margin-bottom:20px}}
  .tile {{background:#fff;border:1px solid #dee2e6;border-radius:6px;padding:14px 20px;min-width:140px}}
  .tile .num {{font-size:28px;font-weight:700;line-height:1}}
  .tile .label {{font-size:12px;color:#666;margin-top:4px}}
  .tile.warn .num {{color:#856404}}
  .tabs {{display:flex;gap:0;border-bottom:2px solid #dee2e6;margin-bottom:0}}
  .tab {{padding:8px 18px;cursor:pointer;border:1px solid transparent;border-bottom:none;margin-bottom:-2px;background:#f8f9fa;font-size:14px}}
  .tab.active {{background:#fff;border-color:#dee2e6;border-bottom-color:#fff}}
  .panel {{display:none;background:#fff;border:1px solid #dee2e6;border-top:none;padding:16px}}
  .panel.active {{display:block}}
  table {{border-collapse:collapse;width:100%;font-size:13px}}
  th {{background:#f1f3f5;text-align:left;padding:7px 10px;border-bottom:2px solid #dee2e6;white-space:nowrap}}
  td {{padding:6px 10px;border-bottom:1px solid #f1f3f5;vertical-align:top}}
  tr:hover td {{background:rgba(0,0,0,.03)}}
</style>
</head>
<body>
<h1>Deal Monitor</h1>
<div class="tiles">
  <div class="tile">
    <div class="num">{len(new_opps)}</div>
    <div class="label">New Opportunities</div>
  </div>
  <div class="tile{' warn' if needs_attention else ''}">
    <div class="num">{len(active)}</div>
    <div class="label">Active Quoting ({len(needs_attention)} need attention)</div>
  </div>
  <div class="tile">
    <div class="num">{len(closed)}</div>
    <div class="label">Closed</div>
  </div>
</div>
<div class="tabs">
  <div class="tab active" onclick="show('new')">New Opportunities ({len(new_opps)})</div>
  <div class="tab" onclick="show('quoting')">Quoting ({len(active)})</div>
  <div class="tab" onclick="show('closed')">Closed ({len(closed)})</div>
</div>
<div id="new" class="panel active">
  <table>
    <tr><th>Account</th><th>Submitter</th><th>ARR</th><th>Location</th><th>SKUs</th><th>Go-Live</th><th>Score</th><th>Rationale</th><th>Risk Flags</th></tr>
    {new_opps_rows()}
  </table>
</div>
<div id="quoting" class="panel">
  <table>
    <tr><th>Account</th><th>Thread Account</th><th>Pipeline Step</th><th>Sent Date</th><th>Thread</th></tr>
    {quoting_rows(active)}
  </table>
</div>
<div id="closed" class="panel">
  <table>
    <tr><th>Account</th><th>Thread</th></tr>
    {closed_rows()}
  </table>
</div>
<script>
function show(id){{
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  event.target.classList.add('active');
}}
</script>
</body>
</html>"""
    return html


def save_dashboard(html_content: str, output_path: str = DASHBOARD_OUTPUT_PATH) -> str:
    """Write the HTML dashboard to disk. Returns the absolute output path."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return os.path.abspath(output_path)


def generate_summary(
    new_opps: list[Any],
    quoting: list[Any],
    sku_risk_flags: list[str] | None = None,
) -> str:
    """Generate the brief text summary for chat output (SKILL.md Step 6).

    Format: "N new opps · M active quoting (K need attention) · J closed"
    """
    closed = [d for d in quoting if d.step == 6]
    active = [d for d in quoting if d.step != 6]
    needs_attention = [d for d in active if d.step in NEEDS_ATTENTION_STEPS]
    risk_note = f" ({len(sku_risk_flags)} unknown-SKU risk)" if sku_risk_flags else ""

    return (
        f"**Deal Monitor** — {len(new_opps)} new opp{'s' if len(new_opps) != 1 else ''}{risk_note}"
        f" · {len(active)} active quoting ({len(needs_attention)} need attention)"
        f" · {len(closed)} closed"
    )
