"""Guard CITATION.cff against version drift and schema rot.

``CITATION.cff`` carries its own ``version`` field, which puts it in the
same trap the plugin manifests fell into (0.3.0 vs 0.6.0, because
nothing checked). Pinning it to ``rote.__version__`` means a release
that forgets the citation file fails CI instead of telling the world to
cite a version that never shipped.

The shape assertions cover the fields GitHub's "Cite this repository"
widget and Zenodo actually read, so a well-formed but useless file
fails here rather than silently rendering wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rote import __version__

REPO_ROOT = Path(__file__).resolve().parent.parent
CITATION = REPO_ROOT / "CITATION.cff"


def _citation() -> dict[str, Any]:
    data = yaml.safe_load(CITATION.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_citation_version_matches_package() -> None:
    data = _citation()
    assert data["version"] == __version__, (
        f"CITATION.cff version {data['version']!r} != rote.__version__ "
        f"{__version__!r}. Bump CITATION.cff on release."
    )


def test_citation_declares_required_fields() -> None:
    data = _citation()
    assert data["cff-version"] == "1.2.0"
    assert data["type"] == "software"
    assert data["title"] == "rote"
    assert data["license"] == "Apache-2.0"
    assert data["authors"], "citation needs at least one author"


def test_citation_references_the_compiled_ai_paper() -> None:
    """The README's headline numbers come from this paper.

    If the reference is dropped, anyone citing rote loses the pointer to
    the evidence the claims rest on.
    """
    refs = _citation()["references"]
    paper = next(r for r in refs if r["type"] == "article")
    assert "arxiv.org/abs/2604.05150" in paper["url"]
    assert paper["authors"][0]["family-names"] == "Trooskens"
