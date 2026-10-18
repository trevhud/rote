"""
Aggregate per-thread email classifications into Dock Compliance counts.

Pure function — takes the list of per-thread classification results from
classify_dock_email (llm_judge, fan_out) and produces approval/request counts.

No third-party imports — stdlib only.
"""

from typing import Any


def aggregate_dock_emails(classified_threads: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Count approved vs. requested dock appointments; surface outstanding requests.

    Input: list of classification results from classify_dock_email (one per thread).
    Each element has:
        {
            "classification": "appointment_approved" | "appointment_requested" | "other",
            "appointment_ref": str | None,   # e.g. appointment ID or subject
            "thread_id": str,
        }

    Returns:
        {
            "approved_count": int,
            "requested_count": int,
            "outstanding_requests": [str, ...],   # refs of unmatched requests
        }
    """
    approved_refs: set[str] = set()
    requested_refs: list[tuple[str, str]] = []  # (appointment_ref, thread_id)

    for item in classified_threads:
        classification = item.get("classification", "other")
        ref = str(item.get("appointment_ref") or item.get("thread_id", ""))

        if classification == "appointment_approved":
            if ref:
                approved_refs.add(ref)
        elif classification == "appointment_requested":
            requested_refs.append((ref, item.get("thread_id", "")))

    outstanding = [
        ref for ref, _tid in requested_refs if ref not in approved_refs
    ]

    return {
        "approved_count": len(approved_refs),
        "requested_count": len(requested_refs),
        "outstanding_requests": outstanding,
    }
