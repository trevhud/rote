"""``python -m rote`` — the same CLI as the ``rote`` console script.

Exists so generated configs can invoke rote through a known
interpreter (``sys.executable -m rote``) instead of guessing at PATH —
the eval harness's ``headersHelper`` wiring depends on this.
"""

import sys

from rote.cli import main

sys.exit(main())
