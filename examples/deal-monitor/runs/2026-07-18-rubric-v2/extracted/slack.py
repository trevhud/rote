"""
Slack intake pull for deal-monitor.

MCP tool: slack_read_channel (slack MCP server)
Underlying API: GET /conversations.history (Slack Web API)
"""
from __future__ import annotations

SLACK_CHANNEL_ID: str = "C0EXAMPLE000"
SLACK_MESSAGE_LIMIT: int = 50


def fetch_slack_messages(channel_id: str = SLACK_CHANNEL_ID, limit: int = SLACK_MESSAGE_LIMIT) -> list[dict]:
    """
    Read the most recent messages from #deal-intake.

    Returns a list of raw Slack message dicts; callers pass to filter_and_extract_opps.
    Graduated from: SKILL.md > Step 1: Pull #deal-intake Slack messages
    """
    raise NotImplementedError(
        "Wire this to the slack MCP server: tool=slack_read_channel, "
        f"channel={channel_id!r}, limit={limit}"
    )
