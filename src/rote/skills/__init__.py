"""Skills on this machine, found and ranked (:mod:`rote.skills.discovery`).

The wheel also carries the compiler's own skill bundle under this
package's directory (``rote/skills/rote-compile/``, see pyproject.toml);
that is data, not a module.
"""

from rote.skills.discovery import (
    Discovery,
    Readiness,
    RegistryView,
    ServerReadiness,
    Skill,
    SkillRoot,
    discover,
)

__all__ = [
    "Discovery",
    "Readiness",
    "RegistryView",
    "ServerReadiness",
    "Skill",
    "SkillRoot",
    "discover",
]
