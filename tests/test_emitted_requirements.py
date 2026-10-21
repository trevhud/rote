"""An emitted Python app's requirements.txt names what its code imports.

The dbos app emitted none (its README installed `dbos` alone), and the
python app's ignored extracted/: a --backend api module importing the
vendor SDK shipped with no way to install it. Both now derive their vendor
lines from the same import map verification and the bench install from.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rote.adapters import get_adapter
from rote.ir import Pipeline


def _pipeline() -> Pipeline:
    return Pipeline.model_validate(
        {
            "name": "reqs-demo",
            "version": "0.1.0",
            "input": {"type": "In"},
            "nodes": [
                {
                    "id": "pull",
                    "kind": "external_call",
                    "description": "list the user's files",
                    "impl": "extracted/pull.py:pull",
                },
                {
                    "id": "shape",
                    "kind": "pure_function",
                    "description": "shape them",
                    "impl": "extracted/shape.py:shape",
                },
            ],
            "edges": [{"from": "pull", "to": "shape"}],
        }
    )


def _source(tmp_path: Path, pull: str, shape: str = "def shape():\n    return {}\n") -> Path:
    src = tmp_path / "src"
    (src / "extracted").mkdir(parents=True)
    (src / "extracted" / "pull.py").write_text(pull, encoding="utf-8")
    (src / "extracted" / "shape.py").write_text(shape, encoding="utf-8")
    return src


def _lines(app: Path) -> list[str]:
    text = (app / "requirements.txt").read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line and not line.startswith("#")]


_DRIVE = (
    "def pull():\n"
    "    from googleapiclient.discovery import build\n"
    "    from google.oauth2.service_account import Credentials\n"
    "    return {}\n"
)


@pytest.mark.parametrize(("runtime", "head"), [("dbos", ["dbos>=3.3,<4"]), ("python", [])])
def test_vendor_sdks_imported_by_extracted_modules_are_required(
    tmp_path: Path, runtime: str, head: list[str]
) -> None:
    src = _source(tmp_path, _DRIVE)
    app = tmp_path / "app"
    get_adapter(runtime, extracted_source_dir=src, external_backend="api").emit(_pipeline(), app)
    assert _lines(app) == [*head, "google-api-python-client", "google-auth"]


def test_a_preserved_user_edit_is_what_the_requirements_follow(tmp_path: Path) -> None:
    src = _source(tmp_path, "def pull():\n    return {}\n")
    app = tmp_path / "app"
    adapter = get_adapter("dbos", extracted_source_dir=src, external_backend="api")
    adapter.emit(_pipeline(), app)
    assert _lines(app) == ["dbos>=3.3,<4"]
    (app / "extracted" / "pull.py").write_text("import httpx\n\ndef pull():\n    return {}\n")
    adapter.emit(_pipeline(), app)
    assert _lines(app) == ["dbos>=3.3,<4", "httpx"]


def test_an_import_rote_cannot_name_is_a_comment_naming_the_file(tmp_path: Path) -> None:
    src = _source(tmp_path, "import stripe_like_sdk\n\ndef pull():\n    return {}\n")
    app = tmp_path / "app"
    get_adapter("dbos", extracted_source_dir=src, external_backend="api").emit(_pipeline(), app)
    text = (app / "requirements.txt").read_text(encoding="utf-8")
    assert (
        "# extracted/pull.py imports stripe_like_sdk: rote cannot name its package, add it here."
        in text
    )
    assert _lines(app) == ["dbos>=3.3,<4"]


def test_the_mcp_helpers_own_imports_are_not_vendor_requirements(tmp_path: Path) -> None:
    pipeline = Pipeline.model_validate(
        {
            "name": "mcp-demo",
            "version": "0.1.0",
            "input": {"type": "In"},
            "nodes": [
                {
                    "id": "pull",
                    "kind": "external_call",
                    "description": "pull",
                    "mcp": {"server": "vendor", "tool": "get_data"},
                }
            ],
            "edges": [],
        }
    )
    app = tmp_path / "app"
    get_adapter("dbos").emit(pipeline, app)
    text = (app / "requirements.txt").read_text(encoding="utf-8")
    assert _lines(app) == ["dbos>=3.3,<4", "fastmcp>=4.1.1,<5"]
    assert "_rote_mcp.py imports" not in text


@pytest.mark.parametrize(
    ("module_import", "pin"),
    [
        ("import pydantic", "pydantic>=2.7"),
        ("from anthropic import Anthropic", "anthropic>=1.15"),
        ("import openai", "openai>=1.40"),
    ],
)
@pytest.mark.parametrize("runtime", ["dbos", "python", "temporal"])
def test_a_judgeless_module_importing_an_sdk_gets_its_pin(
    tmp_path: Path, runtime: str, module_import: str, pin: str
) -> None:
    """The IR has no judge, so only the scan of extracted/ can see the SDK."""
    src = _source(tmp_path, f"{module_import}\n\ndef pull():\n    return {{}}\n")
    app = tmp_path / "app"
    get_adapter(runtime, extracted_source_dir=src, external_backend="api").emit(_pipeline(), app)
    assert pin in _lines(app)
