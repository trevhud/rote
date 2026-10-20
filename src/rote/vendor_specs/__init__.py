"""Vendor API specs: resolve the machine-readable spec a vendor publishes, and
check generated vendor requests against it in code (``--backend api``).

The order is the compile's: schema first (:mod:`.google`, :mod:`.store`),
then code (:mod:`.extract`, :mod:`.check`); a model is never asked what a
spec already says.
"""

from rote.vendor_specs.check import RequestCheck, check_deliverable, report_section, spec_findings
from rote.vendor_specs.model import ApiSpec, Operation, ParamSpec
from rote.vendor_specs.store import SpecStore, SpecUnavailable

__all__ = [
    "ApiSpec",
    "Operation",
    "ParamSpec",
    "RequestCheck",
    "SpecStore",
    "SpecUnavailable",
    "check_deliverable",
    "report_section",
    "spec_findings",
]
