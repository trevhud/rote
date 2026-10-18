"""
Audit report file persistence for the expense audit pipeline.

Saves the rendered markdown report to the finance shared folder.
Payroll reads from this file; the skill mandates it is saved before
the audit is considered complete ("Payroll only pays from a report
marked final").

Dependencies: none (stdlib only — pathlib for file I/O).
"""

from __future__ import annotations

from pathlib import Path


def save_audit_report(
    period: str,
    report_markdown: str,
    finance_folder_path: str | None,
) -> dict[str, str | None]:
    """
    Save the audit report to the finance shared folder.

    Filename convention (per SKILL.md Step 7): ``<period>-expense-audit.md``
    Example: ``2026-09-expense-audit.md``

    Args:
        period: YYYY-MM string, used to construct the filename.
        report_markdown: rendered markdown string from render_audit_report.
        finance_folder_path: absolute or relative path to the finance shared
            folder. When None, the save is skipped and file_path is None.
            The caller (pipeline input) supplies this so the workflow is
            portable across environments.

    Returns:
        {"file_path": str | None}
        Absolute path of the written file, or None when finance_folder_path
        was None (skip mode).

    Raises:
        RuntimeError: if the finance_folder_path directory does not exist or
            is not writable. Does NOT create parent directories — the folder
            must already exist (finance controls it).
    """
    raise NotImplementedError(
        "implement: if finance_folder_path is None return {'file_path': None}; "
        "otherwise path = Path(finance_folder_path) / f'{period}-expense-audit.md'; "
        "verify path.parent.exists(); path.write_text(report_markdown, encoding='utf-8'); "
        "return {'file_path': str(path.resolve())}."
    )
