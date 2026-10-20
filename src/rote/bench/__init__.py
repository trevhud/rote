"""``rote bench``: a scored compile benchmark (a dev tool, and an eval).

It compiles the repo's example skills with the real compiler (this
spends tokens, so it is never part of the automated test suite), scores
each compiled ``pipeline.yaml`` against the example's hand-written
``expected/pipeline.yaml``, and writes a diffable JSON result. See
``docs/compile-bench.md``.

Layout: :mod:`rote.bench.score` is the pure structural scorer (unit
tested), :mod:`rote.bench.collect` reads one compile's artifacts,
:mod:`rote.bench.report` flattens, aggregates and diffs, and
:mod:`rote.bench.runner` runs the jobs and owns the command line.
"""
