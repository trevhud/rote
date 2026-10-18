"""
Typed signature for classify_dock_email — LLM judge that classifies a single
Gmail dock-activity thread as an appointment request, approval, or other.

Used by the ops-report pipeline's classify_dock_email node (fan_out: true).
The node dispatches once per thread from fetch_dock_emails.output.threads.

Input model field names = node inputs: keys (one field: thread).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict


# ── enums ─────────────────────────────────────────────────────────────────────

class AppointmentEmailType(str, Enum):
    APPOINTMENT_APPROVED = "appointment_approved"
    APPOINTMENT_REQUESTED = "appointment_requested"
    OTHER = "other"


# ── models ────────────────────────────────────────────────────────────────────

class ClassifyDockEmailInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread: dict[str, Any]
    """
    A single Gmail thread object returned by search_threads.
    Expected to contain at minimum: thread_id (str), subject (str),
    and messages (list of message objects with body/snippet).
    """


class ClassifyDockEmailOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classification: AppointmentEmailType
    """Whether this thread is an appointment approval, request, or unrelated."""

    appointment_ref: str | None = None
    """
    An identifier for the appointment — typically the appointment ID,
    confirmation number, or subject line — so the aggregator can match
    approvals to requests. None when classification is OTHER.
    """

    thread_id: str
    """Echo of the input thread's thread_id for downstream aggregation."""


# ── signature class ───────────────────────────────────────────────────────────

class ClassifyDockEmail:
    """
    Classify a single dock-activity Gmail thread.

    Pre-filter: if the thread subject contains 'approved' (case-insensitive)
    with no 'request' or 'pending', short-circuit as APPOINTMENT_APPROVED.
    If it contains 'request' or 'pending' with no 'approved', short-circuit
    as APPOINTMENT_REQUESTED. Otherwise, dispatch to LLM.

    The LLM handles ambiguous cases (mixed-thread replies, forwarded chains).
    """

    async def forward(self, inputs: ClassifyDockEmailInput) -> ClassifyDockEmailOutput:
        thread = inputs.thread
        thread_id = str(thread.get("thread_id", thread.get("id", "")))
        subject = str(thread.get("subject", "")).lower()

        # Cheap pre-filter on subject line
        has_approved = "approved" in subject or "confirmation" in subject or "confirmed" in subject
        has_requested = "request" in subject or "pending" in subject or "submitted" in subject

        if has_approved and not has_requested:
            return ClassifyDockEmailOutput(
                classification=AppointmentEmailType.APPOINTMENT_APPROVED,
                appointment_ref=thread.get("subject", thread_id),
                thread_id=thread_id,
            )
        if has_requested and not has_approved:
            return ClassifyDockEmailOutput(
                classification=AppointmentEmailType.APPOINTMENT_REQUESTED,
                appointment_ref=thread.get("subject", thread_id),
                thread_id=thread_id,
            )

        # Dispatch to LLM for ambiguous cases
        raise NotImplementedError(
            "LLM dispatch not yet wired — connect via DSPy or BAML. "
            "The pre-filter handles clear approved/requested cases; this path "
            "handles ambiguous threads (mixed replies, re-requests, etc.)."
        )
