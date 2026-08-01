"""Compatibility patches for maps the vendored engine was not exercised on.

The vendored DeliveryBench engine was developed against procgen maps, all of
which ship a full complement of sidecar files and a populated ``bus_routes``
list. CityCore Paris does not, and PLAN.md 3.2 already records why:
``progen_world_enriched.json`` for Paris "has 568 nodes and an empty
``bus_routes`` list".

Where the engine reacts to a missing feature by crashing rather than by having
none of that feature, that is a defect, and PLAN.md 3.3 lists exactly this class
of Paris gap as work to be done. Following ADR-0003 these are fixed as recorded
runtime patches in our layer; ``vendor/`` is never edited, and each patch
carries the sha256 of the source it replaced so a vendor bump surfaces as a
decision instead of a silent revert.

These are deliberately *not* in ``determinism.py``. A determinism patch changes
which of several correct answers you get; a compatibility patch is the
difference between running and raising. Keeping them separate means a report can
say which kind was active.
"""

from __future__ import annotations

import inspect
from typing import Any

from embodiedbench.artifacts.hashing import sha256_bytes
from embodiedbench.baseline.determinism import PatchRecord, PatchSet

EMPTY_BUS_ROUTES_REASON = (
    "vlm_delivery/entities/bus_manager.py init_bus_system() ends with "
    "`self.create_bus(f'bus_{i+1}', list(self.routes.keys())[0])`, indexing the "
    "first route without checking that any route exists. CityCore Paris ships an "
    "empty `bus_routes` list (PLAN.md 3.2), so every Paris reset raises "
    "IndexError before the environment can be used at all. A map with no bus "
    "routes should have no buses, not fail to load. The patch skips bus creation "
    "when there are no routes and is otherwise identical; on any map that does "
    "define routes the behaviour is unchanged. Tracked as PARIS-F1."
)


def _patched_init_bus_system(self, world_data: dict[str, Any]) -> None:
    """``init_bus_system`` that tolerates a map with no bus routes."""
    self._birth_sim_time = float(self.clock.now_sim())
    self.load_routes_from_world_data(world_data)
    route_ids = list(self.routes.keys())
    if not route_ids:
        # No routes on this map: no buses. The bus manager stays valid and
        # empty, so NAVIGATE(mode="bus") reports no service rather than
        # dereferencing a bus that was never created.
        return
    for index in range(self.num_buses):
        self.create_bus(f"bus_{index + 1}", route_ids[0])


def apply_map_compatibility_patches() -> PatchSet:
    """Apply every map-compatibility patch. Idempotent."""
    from embodiedbench.baseline.replay import _ensure_vendor_on_path

    _ensure_vendor_on_path()
    from vagen.envs.deliverybench.vlm_delivery.entities import bus_manager as bus_module

    if getattr(bus_module, "_embodiedbench_compat_applied", False):
        return getattr(bus_module, "_embodiedbench_compat_patchset")

    patches = PatchSet()
    original = bus_module.BusManager.init_bus_system
    source = inspect.getsource(original)
    record = PatchRecord(
        target="vlm_delivery.entities.bus_manager.BusManager.init_bus_system",
        reason=EMPTY_BUS_ROUTES_REASON,
        original_source_sha256=sha256_bytes(source.encode()),
        applied=False,
    )
    if "list(self.routes.keys())[0]" not in source:
        record.skipped_reason = (
            "vendored init_bus_system no longer indexes the first route unguarded; "
            "re-review this patch against the new implementation before applying"
        )
    else:
        bus_module.BusManager.init_bus_system = _patched_init_bus_system
        record.applied = True
    patches.records.append(record)

    bus_module._embodiedbench_compat_applied = True
    bus_module._embodiedbench_compat_patchset = patches
    return patches
