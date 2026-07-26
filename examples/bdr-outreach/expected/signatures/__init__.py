"""Typed LLM signatures for the compiled BDR pipeline.

Each signature in this package is a typed wrapper around a single LLM
call. They replace the agent loops in the source skill where the LLM
was being asked to make a fuzzy classification (vetting, personalization)
that nonetheless has a structured output.

The signatures are deliberately framework-agnostic for v0 — they expose
typed input/output via Pydantic and a stub ``forward()`` method. A
production implementation would back each signature with DSPy
(``dspy.compile`` for prompt + few-shot optimization) or BAML.

The Temporal adapter wraps each signature in an ``@activity.defn``
because LLM calls are non-deterministic and must live in activities,
not workflow code.
"""
