from extracted.parse_dock_activity_emails import parse_dock_activity_emails


def make_threads(*snippets):
    return {"threads": [{"snippet": s, "messages": []} for s in snippets], "resultSizeEstimate": len(snippets)}


# --- approved count ---

def test_approved_thread_counted():
    result = parse_dock_activity_emails(make_threads("Appointment approved for EAST1"))
    assert result["approved_count"] == 1
    assert result["requested_count"] == 0
    assert result["outstanding_requests"] == []


def test_multiple_approved_threads():
    result = parse_dock_activity_emails(make_threads(
        "Appointment approved", "Dock slot approved for Tuesday"
    ))
    assert result["approved_count"] == 2


# --- requested (outstanding) count ---

def test_requested_thread_counted_as_outstanding():
    # skill: "Flag any outstanding requests not yet approved"
    result = parse_dock_activity_emails(make_threads("Appointment requested for WEST2"))
    assert result["requested_count"] == 1
    assert len(result["outstanding_requests"]) == 1


def test_pending_thread_counted_as_outstanding():
    result = parse_dock_activity_emails(make_threads("Dock slot pending review"))
    assert result["requested_count"] == 1


# --- approved supersedes requested ---

def test_approved_thread_not_counted_as_outstanding():
    result = parse_dock_activity_emails(make_threads("Request approved for EAST1"))
    assert result["approved_count"] == 1
    assert result["requested_count"] == 0
    assert result["outstanding_requests"] == []


# --- neither keyword ---

def test_unrelated_thread_not_counted():
    result = parse_dock_activity_emails(make_threads("Weekly newsletter"))
    assert result["approved_count"] == 0
    assert result["requested_count"] == 0


# --- empty input ---

def test_empty_threads_returns_zeros():
    result = parse_dock_activity_emails({"threads": []})
    assert result == {
        "approved_count": 0,
        "requested_count": 0,
        "outstanding_requests": [],
        "total_threads": 0,
    }


def test_missing_threads_key_returns_zeros():
    result = parse_dock_activity_emails({})
    assert result["total_threads"] == 0


# --- total threads ---

def test_total_threads_count():
    result = parse_dock_activity_emails(make_threads(
        "approved", "requested", "weekly notes"
    ))
    assert result["total_threads"] == 3


# --- subject-line matching (message headers) ---

def test_subject_header_approved():
    threads = {
        "threads": [{
            "snippet": "see attached",
            "messages": [{
                "payload": {
                    "headers": [{"name": "Subject", "value": "Dock Appointment Approved"}]
                }
            }]
        }]
    }
    result = parse_dock_activity_emails(threads)
    assert result["approved_count"] == 1
