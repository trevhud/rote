# Compile report: ops-report

rote wrote this report from `pipeline.yaml` and the extracted code. Each node's description is the compiler's reason for its kind; each `# skill: "..."` citation is the rule a line of code implements; each `# rote: assumption:` marker is a decision the skill does not state.

Compiled with System-1: rote decided the structure with a structure pass: claude-haiku-5-5 via claude -p (subscription) read the whole skill and proposed every node in 1 round(s), and code checked each one against the skill's text, its tools' schemas and the sample input; no decision model checked it (--decider rules), and the agent wrote the implementations and judge prompts. Every decision, with its confidence and rule, is in `system1/decisions.json`. Its model calls cost $0.0060 (claude-haiku-5-5, structure pass), on top of the agent's run.

## Summary

- 11 node(s): 6 `pure_function`, 4 `external_call`, 0 `llm_judge`, 0 `agent_loop`, 1 `hitl_gate`.
- Codified (`pure_function` + `external_call`): 10 of 10 non-gate node(s), 100%.
- Implemented impl node(s): 6 of 10.
- Entry: `shipment_containers_read`, `dock_pending_read`, `dock_emails_search`, `open_dwell_read`.
- Exit: `executive_report`.
- Mandatory: `shipment_containers_read`, `shipment_containers_summary`, `dock_pending_read`, `dock_pending_summary`, `dock_emails_search`, `dock_email_summary`, `open_dwell_read`, `open_dwell_filter`, `manual_data_flags`, `executive_report`.
- HITL gate `manual_data_gate` (signal `manual_data`, timeout 7d): Manual data to request from user:: Pause until the duty manager supplies the dock appointment, sorter, and current dwell numbers that live behind dashboards.
- MCP servers: `gmail` (dock_emails_search), `google_drive` (dock_pending_read, open_dwell_read, shipment_containers_read).

## Nodes

| Node | Kind | Implementation | Skill section | Description |
|---|---|---|---|---|
| `shipment_containers_read` | external_call | `extracted/shipment_containers_read.py:shipment_containers_read` (stub) | Data sources to auto-pull: | Reads the shipment containers spreadsheet from Google Drive and returns its first tab (the facility summary table with a COMPLETION % column) as CSV text. |
| `shipment_containers_summary` | pure_function | `extracted/shipment_containers_summary.py:shipment_containers_summary` (implemented) | Data sources to auto-pull: | Parses the shipment container facility summary into per-site active, loaded, staged, shipped counts and completion %, and flags sites below the completion threshold. |
| `dock_pending_read` | external_call | `extracted/dock_pending_read.py:dock_pending_read` (stub) | Data sources to auto-pull: | Reads the Dock Pending Log spreadsheet from Google Drive and returns its content as CSV text. |
| `dock_pending_summary` | pure_function | `extracted/dock_pending_summary.py:dock_pending_summary` (implemented) | Data sources to auto-pull: | Counts the open dock pending log rows where Resolved = FALSE, in total, by site and by issue type. |
| `dock_emails_search` | external_call | `extracted/dock_emails_search.py:dock_emails_search` (stub) | Data sources to auto-pull: | Searches Gmail for dock-activity threads from the last day and returns the matching threads as an array. |
| `dock_email_summary` | pure_function | `extracted/dock_email_summary.py:dock_email_summary` (implemented) | Data sources to auto-pull: | Counts Approved versus Requested dock appointments from the dock-activity email threads and lists the outstanding requests not yet approved. |
| `open_dwell_read` | external_call | `extracted/open_dwell_read.py:open_dwell_read` (stub) | Data sources to auto-pull: | Reads the dwell ticket log tab "Dwell Log - Form Responses" from Google Drive and returns its content as CSV text. |
| `open_dwell_filter` | pure_function | `extracted/open_dwell_filter.py:open_dwell_filter` (implemented) | Data sources to auto-pull: | Filters the dwell ticket log to rows whose Open/Closed column is "Open", reports each ticket's details, and flags missort and misload thresholds. |
| `manual_data_gate` | hitl_gate | signal `manual_data` | Manual data to request from user: | Manual data to request from user:: Pause until the duty manager supplies the dock appointment, sorter, and current dwell numbers that live behind dashboards. |
| `manual_data_flags` | pure_function | `extracted/manual_data_flags.py:manual_data_flags` (implemented) | Manual data to request from user: | Applies the skill's manual-data rules to the duty manager's inputs, giving appointment escalations, sorter flags and current dwell alerts. |
| `executive_report` | pure_function | `extracted/executive_report.py:executive_report` (implemented) | Report format: | Assembles the executive operations report with a KPI header, three sections, and a reminder to open the live "ops-report" artifact. |

## Crystallization log

- `extracted/dock_email_summary.py` (2 skill citation(s))
  - line 53: "Count Approved vs. Requested appointments."
  - line 57: "Flag any outstanding requests not yet approved."
- `extracted/dock_emails_search.py`: Data sources to auto-pull:: direct-API backend stub. (0 skill citation(s))
- `extracted/dock_pending_read.py`: Data sources to auto-pull:: direct-API backend stub. (0 skill citation(s))
- `extracted/dock_pending_summary.py` (1 skill citation(s))
  - line 74: "Count all rows where Resolved = FALSE"
- `extracted/executive_report.py` (2 skill citation(s))
  - line 18: "KPI header: Active sites, Dock Pending count, Avg Completion %, Open Dwell Tickets"
  - line 38: "Also remind the duty manager to open the live artifact"
- `extracted/manual_data_flags.py` (6 skill citation(s))
  - line 41: "and any other color (Non-Compliant). Non-Compliant must be escalated."
  - line 57: "Flag if fail rate > 5% (action needed) or if volume dropped >15% day-over-day"
  - line 62: "Flag if fail rate > 5% (action needed)"
  - line 65: "or if volume dropped >15% day-over-day (divert adjustment may be needed)"
  - line 88: "Brand 25-40 = Missort Warning (light red), 41+ = Missort Alert (bright red)."
  - line 95: "Carrier 80+ missing = Possible Misload Alert (darkest red)."
- `extracted/open_dwell_filter.py` (2 skill citation(s))
  - line 89: "Filter rows where Open/Closed column = "Open"."
  - line 109: "Flag packages >= 25 (Missort Warning), >= 41 (Missort Alert), carrier >= 80 (Possible Misload Alert)."
- `extracted/open_dwell_read.py`: Data sources to auto-pull:: direct-API backend stub. (0 skill citation(s))
- `extracted/shipment_containers_read.py`: Data sources to auto-pull:: direct-API backend stub. (0 skill citation(s))
- `extracted/shipment_containers_summary.py` (1 skill citation(s))
  - line 123: "Flag any site below 75% completion."

## Open questions

### OQ-1: dock_pending_summary

> blank site or issue type is grouped under "Unknown"

Marked `# rote: assumption:` at `extracted/dock_pending_summary.py:78`.

### Structure pass changed a decision: manual_data_gate

> structure pass: `manual_data_gate`: signal 'manual_data_provided' renamed to the sample's 'manual_data'

### Contract: shipment_containers_read

> result shape of google_drive.read_file_content is assumed, not recorded; downstream nodes read the whole result. Run `rote baseline` against the real server to record it

### Contract: dock_pending_read

> result shape of google_drive.read_file_content is assumed, not recorded; downstream nodes read the whole result. Run `rote baseline` against the real server to record it

### Contract: dock_emails_search

> result shape of gmail.search_threads is assumed, not recorded; downstream nodes read the whole result. Run `rote baseline` against the real server to record it

### Contract: open_dwell_read

> result shape of google_drive.read_file_content is assumed, not recorded; downstream nodes read the whole result. Run `rote baseline` against the real server to record it


## Suggested next steps

- Review every Open question above; each assumption is a rule the skill does not state, and each low-confidence decision took a conservative default.
- Run the pipeline once (`rote run`) and compare it with a run of the skill.
- Recompile with `rote compile` after the skill changes. The skeleton is a model's answer (claude-haiku-5-5's structure pass), checked by code, so two compiles of one skill can differ: compare `system1/skeleton.yaml` between runs before trusting a diff.

## rote verification

- Generated tests: passed (38 passed in 0.06s), run in an OS sandbox.
- Implemented impl nodes: 6 of 10.
- Unresolved problems: 0.
- Assumptions the skill does not state: 1. Review each one:
  - dock_pending_summary:
    - `extracted/dock_pending_summary.py:78` blank site or issue type is grouped under "Unknown"
- Deterministic prefix on the sample input: 2 of 11 node(s) ran on the sample input (4 no recording, 5 not reached).
- Tool schemas: gmail from the skill's bundled snapshot, google_drive from the skill's bundled snapshot.
