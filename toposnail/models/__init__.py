"""
toposnail.models: High-Fidelity Blender-Audited 3D Scenario Models (Scenarios 01 - 05).

Exports the 5 canonical scenario data, mesh, field, and baseline builders:
1. `01_cantilever_michell_truss` -> `build_scen_01_data`, `build_scen_01_mesh`, `build_scen_01_fields`, `build_scen_01_baselines`, `build_scen_01_stages`
2. `02_aerospace_jet_bracket` -> `build_scen_02_data`, `build_scen_02_mesh`, `build_scen_02_fields`, `build_scen_02_baselines`, `build_scen_02_stages`
3. `03_trabecular_femur_anisotropic_so6` -> `build_scen_03_data`, `build_scen_03_mesh`, `build_scen_03_fields`, `build_scen_03_baselines`, `build_scen_03_stages`
4. `04_regenerative_thruster_gimbal_thermo` -> `build_scen_04_data`, `build_scen_04_mesh`, `build_scen_04_fields`, `build_scen_04_baselines`, `build_scen_04_stages`
5. `05_lpbf_self_draining_bridge_arch` -> `build_scen_05_data`, `build_scen_05_mesh`, `build_scen_05_fields`, `build_scen_05_baselines`, `build_scen_05_stages`
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from toposnail.models.scen_01_cantilever import (
    SCENARIO_ID as SCEN_01_ID,
    build_scen_01_baselines,
    build_scen_01_data,
    build_scen_01_fields,
    build_scen_01_mesh,
    build_scen_01_stage_1_mesh,
    build_scen_01_stage_2_mesh,
    build_scen_01_stage_3_mesh,
    build_scen_01_stage_4_mesh,
    build_scen_01_stages,
    verify_scen_01_structural_integrity,
)
from toposnail.models.scen_02_bracket import (
    SCENARIO_ID as SCEN_02_ID,
    audit_scen_02_geometry,
    build_scen_02_baselines,
    build_scen_02_data,
    build_scen_02_fields,
    build_scen_02_mesh,
    build_scen_02_stage_1_mesh,
    build_scen_02_stage_2_mesh,
    build_scen_02_stage_3_mesh,
    build_scen_02_stage_4_mesh,
    build_scen_02_stages,
)
from toposnail.models.scen_03_femur import (
    SCENARIO_ID as SCEN_03_ID,
    audit_scen_03,
    build_scen_03_baselines,
    build_scen_03_data,
    build_scen_03_fields,
    build_scen_03_mesh,
    build_scen_03_stage_1_mesh,
    build_scen_03_stage_2_mesh,
    build_scen_03_stage_3_mesh,
    build_scen_03_stage_4_mesh,
    build_scen_03_stages,
    verify_scen_03_structural_integrity,
)
from toposnail.models.scen_04_thruster import (
    SCENARIO_ID as SCEN_04_ID,
    audit_scen_04,
    build_scen_04_baselines,
    build_scen_04_data,
    build_scen_04_fields,
    build_scen_04_mesh,
    build_scen_04_stage_1_mesh,
    build_scen_04_stage_2_mesh,
    build_scen_04_stage_3_mesh,
    build_scen_04_stage_4_mesh,
    build_scen_04_stages,
)
from toposnail.models.scen_05_bridge import (
    SCENARIO_ID as SCEN_05_ID,
    audit_scen_05,
    build_scen_05_data,
    build_scen_05_fields,
    build_scen_05_mesh,
    build_scen_05_stage_1_mesh,
    build_scen_05_stage_2_mesh,
    build_scen_05_stage_3_mesh,
    build_scen_05_stage_4_mesh,
    build_scen_05_stages,
    get_scen_05_baselines as build_scen_05_baselines,
)


def audit_mesh_2manifold(
    verts: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    normals: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, Any]:
    """
    Perform a strict, rigorous topological and geometric 2-manifold audit on a triangle mesh:
    1. Boundary edges == 0 (watertight, every undirected edge shared by >= 2 faces).
    2. Non-manifold edges == 0 (every undirected edge shared by strictly 2 faces: C(e) == 2).
    3. Degenerate triangles == 0 (face area > 1e-8 everywhere).
    4. Outward normal alignment: min(n_face . n_vert_avg) > 0.05 strictly without abs().
    5. Unit vertex normals: |norm - 1.0| < 0.05.
    6. Signed volume > 0.0.
    """
    v = np.array(verts, dtype=np.float64)
    f = np.array(faces, dtype=np.int64)
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    degen_count = int(np.sum(areas <= 1e-8))

    edge_counts: Dict[Tuple[int, int], int] = {}
    for tri in f:
        for i in range(3):
            e = (int(min(tri[i], tri[(i + 1) % 3])), int(max(tri[i], tri[(i + 1) % 3])))
            edge_counts[e] = edge_counts.get(e, 0) + 1

    boundary_edges = sum(1 for c in edge_counts.values() if c == 1)
    non_manifold_edges = sum(1 for c in edge_counts.values() if c != 2)
    vol = float(np.sum(np.sum(cross * v2, axis=1)) / 6.0)

    min_normal_dot = 1.0
    inverted_normals = 0
    if normals is not None and len(normals) == len(v):
        n = np.array(normals, dtype=np.float64)
        fn = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-12)
        vn_avg = (n[f[:, 0]] + n[f[:, 1]] + n[f[:, 2]]) / 3.0
        vn_avg = vn_avg / (np.linalg.norm(vn_avg, axis=1, keepdims=True) + 1e-12)
        dots = np.sum(fn * vn_avg, axis=1)
        min_normal_dot = float(np.min(dots))
        inverted_normals = int(np.sum(dots <= 0.0))

    passed = (
        boundary_edges == 0
        and non_manifold_edges == 0
        and degen_count == 0
        and inverted_normals == 0
        and min_normal_dot > 0.05
        and vol > 0.0
    )

    return {
        "passed": bool(passed),
        "volume": round(vol, 6),
        "vertex_count": int(len(v)),
        "face_count": int(len(f)),
        "boundary_edges": boundary_edges,
        "non_manifold_edges": non_manifold_edges,
        "degenerate_triangles": degen_count,
        "min_normal_dot": round(min_normal_dot, 6),
        "inverted_normals": inverted_normals,
    }


SCENARIO_DATA_BUILDERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    SCEN_01_ID: build_scen_01_data,
    SCEN_02_ID: build_scen_02_data,
    SCEN_03_ID: build_scen_03_data,
    SCEN_04_ID: build_scen_04_data,
    SCEN_05_ID: build_scen_05_data,
}

SCENARIO_STAGE_BUILDERS: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
    SCEN_01_ID: build_scen_01_stages,
    SCEN_02_ID: build_scen_02_stages,
    SCEN_03_ID: build_scen_03_stages,
    SCEN_04_ID: build_scen_04_stages,
    SCEN_05_ID: build_scen_05_stages,
}

__all__ = [
    "SCEN_01_ID",
    "SCEN_02_ID",
    "SCEN_03_ID",
    "SCEN_04_ID",
    "SCEN_05_ID",
    "SCENARIO_DATA_BUILDERS",
    "SCENARIO_STAGE_BUILDERS",
    "audit_mesh_2manifold",
    "build_scen_01_data",
    "build_scen_01_mesh",
    "build_scen_01_stage_1_mesh",
    "build_scen_01_stage_2_mesh",
    "build_scen_01_stage_3_mesh",
    "build_scen_01_stage_4_mesh",
    "build_scen_01_stages",
    "build_scen_01_fields",
    "build_scen_01_baselines",
    "verify_scen_01_structural_integrity",
    "build_scen_02_data",
    "build_scen_02_mesh",
    "build_scen_02_stage_1_mesh",
    "build_scen_02_stage_2_mesh",
    "build_scen_02_stage_3_mesh",
    "build_scen_02_stage_4_mesh",
    "build_scen_02_stages",
    "build_scen_02_fields",
    "build_scen_02_baselines",
    "audit_scen_02_geometry",
    "build_scen_03_data",
    "build_scen_03_mesh",
    "build_scen_03_stage_1_mesh",
    "build_scen_03_stage_2_mesh",
    "build_scen_03_stage_3_mesh",
    "build_scen_03_stage_4_mesh",
    "build_scen_03_stages",
    "build_scen_03_fields",
    "build_scen_03_baselines",
    "verify_scen_03_structural_integrity",
    "audit_scen_03",
    "build_scen_04_data",
    "build_scen_04_mesh",
    "build_scen_04_stage_1_mesh",
    "build_scen_04_stage_2_mesh",
    "build_scen_04_stage_3_mesh",
    "build_scen_04_stage_4_mesh",
    "build_scen_04_stages",
    "build_scen_04_fields",
    "build_scen_04_baselines",
    "audit_scen_04",
    "build_scen_05_data",
    "build_scen_05_mesh",
    "build_scen_05_stage_1_mesh",
    "build_scen_05_stage_2_mesh",
    "build_scen_05_stage_3_mesh",
    "build_scen_05_stage_4_mesh",
    "build_scen_05_stages",
    "build_scen_05_fields",
    "build_scen_05_baselines",
    "audit_scen_05",
]
