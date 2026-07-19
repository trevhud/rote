# Crystallized from SKILL.md Step 1.
# Vendor API: Slack conversations.history (via slack MCP tool read_channel)
from __future__ import annotations

from dataclasses import dataclass

SLACK_CHANNEL_ID = "C0EXAMPLE000"  # SKILL.md:22 — #deal-intake
SLACK_MESSAGE_LIMIT = 50           # SKILL.md:22 — "50 most recent messages"


@dataclass
class SlackMessage:
    ts: str
    text: str
    user: str


def fetch_deal_intake_messages(channel_id: str = SLACK_CHANNEL_ID) -> list[SlackMessage]:
    """Fetch the most recent messages from the deal-intake Slack channel.

    Calls conversations.history with limit=SLACK_MESSAGE_LIMIT.
    Raises on Slack API errors (rate limit, auth) — caller handles retry.

    NOTE: Implementor must wire the Slack MCP client here.
    POST /api/conversations.history {channel: channel_id, limit: SLACK_MESSAGE_LIMIT}
    """
    raise NotImplementedError(
        "Wire the Slack MCP client: call conversations.history on channel_id "
        "with limit=SLACK_MESSAGE_LIMIT and return the messages list."
    )
