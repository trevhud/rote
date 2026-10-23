"""The terminal layer for onboarding commands: mode, prompts, plans, output.

See ``docs/design/first-run.md``. The rule this package exists for: every
side effect is an :class:`~rote.interact.plan.Action`, shown and confirmed
before it runs, and interactivity is decided once
(:class:`~rote.interact.mode.Mode`) before anything is asked.
"""
