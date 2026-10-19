"""
Parse Gmail search_threads result for dock activity emails.

Assumed input shape: dict with 'threads' key (list of thread dicts).
Each thread has at minimum 'snippet' (str) and optionally 'messages' (list).
The dock-activity label captures appointment request/approval threads from the
past day (query: label:dock-activity newer_than:1d).
"""


def parse_dock_activity_emails(threads_result: dict) -> dict:
    """Count approved vs. requested dock appointments in Gmail dock-activity threads.

    Threads are classified by keyword matching on snippet and subject:
    - A thread whose text contains 'approved' is counted as approved.
    - A thread whose text contains 'requested', 'request', or 'pending'
      (and NOT 'approved') is counted as a requested (outstanding) appointment.

    Args:
        threads_result: Return value from Gmail search_threads MCP call.
            Expected shape: {"threads": list, "resultSizeEstimate": int}

    Returns:
        {
            "approved_count": int,
            "requested_count": int,
            "outstanding_requests": list,   # thread dicts not yet approved
            "total_threads": int,
        }
    """
    # rote: assumption: approved vs. requested classification uses keyword matching
    # on thread snippet/subject. The skill says "Count Approved vs. Requested
    # appointments" but does not specify the classification signal (field name,
    # subject prefix, label, etc.). Keyword matching is the most portable approach
    # without observed MCP traffic.
    threads = threads_result.get("threads") or []
    if not threads:
        return {
            "approved_count": 0,
            "requested_count": 0,
            "outstanding_requests": [],
            "total_threads": 0,
        }

    approved_count = 0
    requested_count = 0
    outstanding_requests = []

    for thread in threads:
        snippet = (thread.get("snippet") or "").lower()
        subject = ""
        messages = thread.get("messages") or []
        if messages:
            headers = (messages[0].get("payload") or {}).get("headers") or []
            for h in headers:
                if (h.get("name") or "").lower() == "subject":
                    subject = (h.get("value") or "").lower()
                    break

        text = snippet + " " + subject

        is_approved = "approved" in text
        # skill: "Flag any outstanding requests not yet approved"
        is_requested = any(kw in text for kw in ("requested", "request", "pending"))

        if is_approved:
            approved_count += 1
        elif is_requested:
            requested_count += 1
            outstanding_requests.append(thread)

    return {
        "approved_count": approved_count,
        "requested_count": requested_count,
        "outstanding_requests": outstanding_requests,
        "total_threads": len(threads),
    }
