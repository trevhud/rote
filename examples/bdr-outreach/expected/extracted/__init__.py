"""Extracted deterministic functions for the graduated BDR pipeline.

Every function in this package was originally invoked as an MCP tool inside
the bdr-outreach Claude skill's agent loop. Graduation extracts them into
deterministic Python functions that call the underlying vendor APIs
directly — no MCP runtime, no LLM, no agent overhead.

The IR's ``external_call`` and ``pure_function`` nodes reference these
functions via their ``impl:`` field. The Temporal adapter wraps each one
in an ``@activity.defn`` for durable execution.
"""
