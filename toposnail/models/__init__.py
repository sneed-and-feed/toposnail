"""
toposnail.models: High-Fidelity Blender-Audited 3D Scenario Models (Scenarios 01 - 05).

Exports the 5 canonical scenario data, mesh, field, and baseline builders:
1. `01_cantilever_michell_truss` -> `build_scen_01_data`, `build_scen_01_mesh`, `build_scen_01_fields`, `build_scen_01_baselines`
2. `02_aerospace_jet_bracket` -> `build_scen_02_data`, `build_scen_02_mesh`, `build_scen_02_fields`, `build_scen_02_baselines`
3. `03_trabecular_femur_anisotropic_so6` -> `build_scen_03_data`, `build_scen_03_mesh`, `build_scen_03_fields`, `build_scen_03_baselines`
4. `04_regenerative_thruster_gimbal_thermo` -> `build_scen_04_data`, `build_scen_04_mesh`, `build_scen_04_fields`, `build_scen_04_baselines`
5. `05_lpbf_self_draining_bridge_arch` -> `build_scen_05_data`, `build_scen_05_mesh`, `build_scen_05_fields`, `build_scen_05_baselines`
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from toposnail.models.scen_01_cantilever import (
    SCENARIO_ID as SCEN_01_ID,
    build_scen_01_baselines,
    build_scen_01_data,
    build_scen_01_fields,
    build_scen_01_mesh,
    verify_scen_01_structural_integrity,
)
from toposnail.models.scen_02_bracket import (
    SCENARIO_ID as SCEN_02_ID,
    audit_scen_02_geometry,
    build_scen_02_baselines,
    build_scen_02_data,
    build_scen_02_fields,
    build_scen_02_mesh,
)
from toposnail.models.scen_03_femur import (
    SCENARIO_ID as SCEN_03_ID,
    audit_scen_03,
    build_scen_03_baselines,
    build_scen_03_data,
    build_scen_03_fields,
    build_scen_03_mesh,
    verify_scen_03_structural_integrity,
)
from toposnail.models.scen_04_thruster import (
    SCENARIO_ID as SCEN_04_ID,
    audit_scen_04,
    build_scen_04_baselines,
    build_scen_04_data,
    build_scen_04_fields,
    build_scen_04_mesh,
)
from toposnail.models.scen_05_bridge import (
    SCENARIO_ID as SCEN_05_ID,
    audit_scen_05,
    build_scen_05_data,
    build_scen_05_fields,
    build_scen_05_mesh,
    get_scen_05_baselines as build_scen_05_baselines,
)

SCENARIO_DATA_BUILDERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    SCEN_01_ID: build_scen_01_data,
    SCEN_02_ID: build_scen_02_data,
    SCEN_03_ID: build_scen_03_data,
    SCEN_04_ID: build_scen_04_data,
    SCEN_05_ID: build_scen_05_data,
}

__all__ = [
    "SCEN_01_ID",
    "SCEN_02_ID",
    "SCEN_03_ID",
    "SCEN_04_ID",
    "SCEN_05_ID",
    "SCENARIO_DATA_BUILDERS",
    "build_scen_01_data",
    "build_scen_01_mesh",
    "build_scen_01_fields",
    "build_scen_01_baselines",
    "verify_scen_01_structural_integrity",
    "build_scen_02_data",
    "build_scen_02_mesh",
    "build_scen_02_fields",
    "build_scen_02_baselines",
    "audit_scen_02_geometry",
    "build_scen_03_data",
    "build_scen_03_mesh",
    "build_scen_03_fields",
    "build_scen_03_baselines",
    "verify_scen_03_structural_integrity",
    "audit_scen_03",
    "build_scen_04_data",
    "build_scen_04_mesh",
    "build_scen_04_fields",
    "build_scen_04_baselines",
    "audit_scen_04",
    "build_scen_05_data",
    "build_scen_05_mesh",
    "build_scen_05_fields",
    "build_scen_05_baselines",
    "audit_scen_05",
]
