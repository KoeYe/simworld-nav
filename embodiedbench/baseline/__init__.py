"""M0 baseline freeze: pin every input the vertical slice is built on.

PLAN.md M0 requires ``BASELINE_MANIFEST.json`` to contain every commit, content,
and plugin hash and to validate in a clean checkout with one non-interactive
command.

Two files implement that while respecting PLAN.md 14.1 ("licensed asset paths
belong in deployment configuration ... not in portable benchmark manifests"):

``BASELINE_MANIFEST.json``
    Tracked and portable. Every location is a symbolic root (``$REPO``,
    ``$CONTENT_STORE``, ``$UE_ROOT``, ``$SPEAR_ROOT``) plus a relative path. It
    carries the digests, licenses, and publishability decisions.

``baseline_roots.local.json``
    Untracked host configuration mapping symbolic roots to absolute paths.
"""

from embodiedbench.baseline.manifest import (
    BaselineManifest,
    ManifestEntry,
    VerificationResult,
    build_manifest,
    load_manifest,
    load_roots,
    verify_manifest,
)

__all__ = [
    "BaselineManifest",
    "ManifestEntry",
    "VerificationResult",
    "build_manifest",
    "load_manifest",
    "load_roots",
    "verify_manifest",
]
