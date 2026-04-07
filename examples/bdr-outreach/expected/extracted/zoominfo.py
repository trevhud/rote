"""ZoomInfo enrichment — batch contact lookup.

# MCP origin

Originally invoked from inside the bdr-outreach skill's lead generation
loop via the ``zoominfo_enrich_contacts`` MCP tool. The skill always
requested ``employmentHistory`` because the LLM vetting step depended on
it.

# Graduated form

After graduation, this is a deterministic call to the ZoomInfo
Enrichment REST endpoint::

    POST https://api.zoominfo.com/enrich/contact
    {
        "matchPersonInput": [...],
        "outputFields": [...]
    }

The batch size is hard-locked at 10 (a ZoomInfo API limit, not a
heuristic) and is enforced by the IR's ``constants.batch_size`` field.
The Temporal activity wrapper applies retry-with-backoff for rate-limit
and network errors.
"""

from __future__ import annotations

from ..types import EnrichedContact, RawContact

# Constant lifted from the IR (constants.batch_size). Kept here for
# defense-in-depth — even if the workflow forgets to pre-batch, this
# function will reject oversize batches rather than silently truncating.
BATCH_SIZE: int = 10

# Constant lifted from the IR (constants.output_fields). Encoded once
# in code so the prompt and the wire payload can never disagree.
OUTPUT_FIELDS: tuple[str, ...] = (
    "firstName",
    "lastName",
    "jobTitle",
    "email",
    "phone",
    "mobilePhone",
    "contactAccuracyScore",
    "companyName",
    "externalUrls",
    "employmentHistory",
    "directPhoneDoNotCall",
    "mobilePhoneDoNotCall",
    "validDate",
)


async def enrich_batch(contacts: list[RawContact]) -> list[EnrichedContact]:
    """Enrich up to 10 contacts in a single ZoomInfo API call.

    Args:
        contacts: A batch of 1–10 raw contacts to enrich.

    Returns:
        EnrichedContact objects in the same order as the input. Contacts
        that ZoomInfo could not match are filtered out by the caller, not
        here, so the index correspondence is preserved.

    Raises:
        ValueError: if the batch exceeds 10 contacts (ZoomInfo API limit).
        NotImplementedError: stub for v0. Production calls
            ``POST https://api.zoominfo.com/enrich/contact``.
    """
    if len(contacts) > BATCH_SIZE:
        raise ValueError(
            f"ZoomInfo enrich batch size is {BATCH_SIZE} (got {len(contacts)})"
        )
    raise NotImplementedError(
        "zoominfo.enrich_batch: implement against ZoomInfo Enrichment API"
    )
