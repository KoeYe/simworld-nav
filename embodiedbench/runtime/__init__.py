"""Runtime implementations behind one API (PLAN.md 7).

``text``, ``cached``, and ``live`` share transition semantics and differ only in
declared sensor capability. Only ``text`` exists at M1.
"""

from embodiedbench.runtime.core import EmbodiedRuntime, RuntimeError_, SnapshotUnsupported

__all__ = ["EmbodiedRuntime", "RuntimeError_", "SnapshotUnsupported"]
