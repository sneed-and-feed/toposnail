"""
tests/test_blind_cad_benchmark.py: Comprehensive Pytest Suite for the TopoSnail
Reference-Free / Zero-Template Blind CAD Ingestion & Intrinsic Physics/Manifold Benchmark.

Verifies:
1. Cold .STL (binary & ASCII) and .OBJ mesh ingestion with vertex welding and outward normal audit.
2. Automatic reference-free bore (Omega_KOZ), mounting boss (Omega_solid), and boundary condition
   recognition with mutually exclusive 3-zone partitioning.
3. All 8 Reference-Free Intrinsic Physical, Geometric & Topological Laws across the 5-part blind
   CAD corpus:
   - Law 1: `stage1_self_baseline_stiffness_efficiency` (eta_stiff > 1.35)
   - Law 2: `bore_clearance_survival_pct` (== 100.0%, 0 intruding voxels/triangles)
   - Law 3: `boss_datum_retention_pct` (>= 99.0%, 0.000 datum erosion)
   - Law 4: `load_path_beta0_connected` (beta_0 == 1 & load_to_support_connected == True)
   - Law 5: `true_equilibrium_residual_admissible` (r_eq <= 5.0e-2)
   - Law 6: `strict_4stage_volume_monotonicity` (V_S1 > V_S2 > V_S3 > V_S4 & |V_S4/V_S1 - V_target| <= 0.015)
   - Law 7: `intrinsic_voxel_quality` (beta_2 == 0, 0.0% checkerboards, >= 98.0% binary crispness)
   - Law 8: `watertight_2manifold_stl_closure` (boundary_edges == 0, non_manifold_edges == 0,
     0 degenerate faces, min_dot > 0.05, divergence_volume > 0, and beta_1(S4) >= beta_1(S1))
4. 4-Solver Blind CAD Comparison (`TopoSnail` vs `UnfilteredSIMP` vs `GaussianFilteredSIMP` vs `HeavisideProjectedSIMP`).
5. CLI Runner (`scripts/run_blind_cad_benchmark.py`), JSON summary, and 5x4 PNG visual audit sheet.
"""

from __future__ import annotations

import json
from pathlib import Path
import struct
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from scripts.run_blind_cad_benchmark import main as cli_main
from toposnail.blind_cad_benchmark import (
    _extract_progressive_blind_cad_stage_mesh,
    _fallback_generate_blind_cad_corpus,
    audit_blind_cad_part_intrinsic,
    ensure_blind_cad_corpus,
    format_blind_cad_benchmark_tables,
    generate_blind_cad_audit_sheet,
    load_blind_cad_file,
    make_annular_cylinder,
    recognize_features_from_blind_cad,
    run_4solver_blind_cad_comparison,
    run_blind_cad_corpus_benchmark,
    run_blind_cad_part_benchmark,
    run_progressive_4stage_blind_cad,
)
from toposnail.io import export_binary_stl, export_obj_mesh


@pytest.fixture(scope="module")
def blind_corpus_manifest(tmp_path_factory):
    """Locate or generate the 5-part blind CAD corpus (.STL & .OBJ) once per module."""
    stl_files = ensure_blind_cad_corpus()
    manifest = []
    for stl_p in stl_files:
        obj_p = stl_p.with_suffix(".obj")
        if not obj_p.exists():
            c_dir = tmp_path_factory.mktemp("blind_corpus_shared")
            stl_files = ensure_blind_cad_corpus(c_dir)
            break
    for stl_p in stl_files:
        obj_p = stl_p.with_suffix(".obj")
        if not obj_p.exists():
            c_dir = tmp_path_factory.mktemp("blind_corpus_fallback")
            return _fallback_generate_blind_cad_corpus(c_dir)
        manifest.append({
            "part_id": stl_p.stem,
            "stl_path": str(stl_p.resolve()),
            "obj_path": str(obj_p.resolve()),
        })
    return manifest


@pytest.fixture(scope="module")
def corpus_results(blind_corpus_manifest, tmp_path_factory):
    """Run the 5-part reference-free blind CAD benchmark once per module."""
    exp_dir = tmp_path_factory.mktemp("blind_exports_shared")
    stl_files = [Path(item["stl_path"]) for item in blind_corpus_manifest]
    return [
        run_blind_cad_part_benchmark(p, grid_shape=(24, 18, 18), v_budget=0.25, max_iters=15, export_dir=exp_dir)
        for p in stl_files
    ]


class TestColdCADIngestionAndFeatureRecognition:
    """Verify cold .STL and .OBJ ingestion and reference-free bore/boss recognition."""

    def test_cold_binary_stl_and_obj_ingestion(self, blind_corpus_manifest):
        """Cold-loading both binary .STL and Wavefront .OBJ must produce welded 2-manifold meshes."""
        assert len(blind_corpus_manifest) == 5

        first = blind_corpus_manifest[0]
        stl_mesh = load_blind_cad_file(first["stl_path"])
        obj_mesh = load_blind_cad_file(first["obj_path"])

        assert stl_mesh.file_format == "STL"
        assert obj_mesh.file_format == "OBJ"
        assert len(stl_mesh.vertices) > 500
        assert len(stl_mesh.faces) > 1000
        assert len(stl_mesh.vertices) == len(obj_mesh.vertices)
        assert len(stl_mesh.faces) == len(obj_mesh.faces)
        assert stl_mesh.divergence_volume > 0.0
        assert obj_mesh.divergence_volume == pytest.approx(stl_mesh.divergence_volume, rel=1e-3)
        assert stl_mesh.genus_beta1 >= 1

    def test_cold_ascii_stl_ingestion(self, tmp_path: Path):
        """Verify ASCII .STL parsing and vertex welding on an unseen test tetrahedron/box."""
        ascii_stl = tmp_path / "unseen_ascii_part.stl"
        ascii_content = """solid unseen_part
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 0 1 0
      vertex 1 1 0
    endloop
  endfacet
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 1 1 0
      vertex 1 0 0
    endloop
  endfacet
endsolid unseen_part
"""
        ascii_stl.write_text(ascii_content, encoding="utf-8")
        loaded = load_blind_cad_file(ascii_stl)
        assert len(loaded.vertices) == 4
        assert len(loaded.faces) == 2

    def test_automatic_bore_and_boss_recognition_3zone_partition(self, corpus_results):
        """
        Verify automatic recognition of cylindrical keep-out bores (Omega_KOZ),
        annular mounting bosses (Omega_solid), and boundary conditions (Omega_fixed, f_ext).
        """
        assert len(corpus_results) == 5

        # Also verify direct recognize_features_from_blind_cad on an alternate grid resolution
        alt_feat = recognize_features_from_blind_cad(
            corpus_results[0].loaded_mesh, grid_shape=(20, 16, 16), v_budget=0.25
        )
        assert alt_feat.to_partition().validate_partition() is True

        for res in corpus_results:
            mesh = res.loaded_mesh
            feat = res.features

            assert len(feat.bores) >= 2, f"{mesh.part_id}: expected >= 2 detected bores"
            assert len(feat.bosses) >= 2, f"{mesh.part_id}: expected >= 2 detected bosses"
            assert np.any(feat.koz_mask), f"{mesh.part_id}: koz_mask must be non-empty"
            assert np.any(feat.solid_mask), f"{mesh.part_id}: solid_mask must be non-empty"
            assert np.any(feat.designable_mask), f"{mesh.part_id}: designable_mask must be non-empty"
            assert np.any(feat.fixed_mask), f"{mesh.part_id}: fixed_mask must be non-empty"
            assert float(np.linalg.norm(feat.forces_3d)) > 0.0, f"{mesh.part_id}: forces_3d must be non-zero"

            # Validate 3-zone mutual exclusivity and completeness
            part = feat.to_partition()
            assert part.validate_partition() is True

    def test_cold_so3_rotated_and_mm_scaled_stl_ingestion(self, blind_corpus_manifest, tmp_path):
        """
        Verify that an unseen binary .STL rotated by an arbitrary 3D rotation matrix R in SO(3),
        scaled to millimeter units (85x), and translated off-origin is automatically aligned in SO(3),
        normalized, feature-recognized, and passes all 8/8 reference-free intrinsic laws.
        """
        src_mesh = load_blind_cad_file(blind_corpus_manifest[1]["stl_path"])  # twin-clevis bellcrank
        # Build non-trivial 3D Euler rotation matrix R in SO(3) (33 deg X, -27 deg Y, 41 deg Z)
        ax, ay, az = np.deg2rad(33.0), np.deg2rad(-27.0), np.deg2rad(41.0)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(ax), -np.sin(ax)], [0.0, np.sin(ax), np.cos(ax)]])
        Ry = np.array([[np.cos(ay), 0.0, np.sin(ay)], [0.0, 1.0, 0.0], [-np.sin(ay), 0.0, np.cos(ay)]])
        Rz = np.array([[np.cos(az), -np.sin(az), 0.0], [np.sin(az), np.cos(az), 0.0], [0.0, 0.0, 1.0]])
        R_so3 = Rz @ Ry @ Rx
        assert np.isclose(np.linalg.det(R_so3), 1.0)

        verts_mm = (src_mesh.vertices.astype(np.float64) @ R_so3.T) * 85.0 + np.array([120.0, -75.0, 210.0])
        anon_stl = tmp_path / "arbitrary_rotated_mm_part.stl"
        export_binary_stl(verts_mm.astype(np.float32), src_mesh.faces, file_path=anon_stl)

        res_rot = run_blind_cad_part_benchmark(
            anon_stl, grid_shape=(24, 18, 18), v_budget=0.25, max_iters=14, export_dir=tmp_path / "rot_stages"
        )
        assert len(res_rot.features.bores) == 3
        assert len(res_rot.features.bosses) == 3
        assert res_rot.features.to_partition().validate_partition() is True
        assert res_rot.intrinsic_audit["all_passed"] is True, f"Rotated mm STL failed laws: {res_rot.intrinsic_audit['laws']}"
        assert res_rot.intrinsic_audit["gates_passed"] == 8

    def test_unseen_5boss_and_oblique_bore_recognition(self, tmp_path: Path):
        """
        Verify that an unseen 5-boss CAD mesh containing an oblique (non-axis-aligned) bore
        preserves its general 3D unit axis vector and prunes redundant support-to-support chords
        so all 8/8 reference-free intrinsic laws pass.
        """
        verts: list[list[float]] = []
        faces: list[list[int]] = []
        colors: list[list[float]] = []
        normals: list[list[float]] = []

        # 4 base mounting bosses along Z and 1 elevated oblique load boss tilted 32 deg in YZ
        base_centers = [
            (-0.52, -0.42, -0.55),
            (-0.52, +0.42, -0.55),
            (+0.52, -0.42, -0.55),
            (+0.52, +0.42, -0.55),
        ]
        for c_b in base_centers:
            make_annular_cylinder(
                verts, faces, colors, normals,
                center=c_b, axis=(0.0, 0.0, 1.0),
                r_inner=0.11, r_outer=0.20, length=0.34,
                stress_outer=0.30, stress_bore=0.40, n_sides=36,
            )
        tilt = np.deg2rad(32.0)
        oblique_axis = (0.0, float(np.cos(tilt)), float(np.sin(tilt)))
        make_annular_cylinder(
            verts, faces, colors, normals,
            center=(0.0, 0.0, 0.55), axis=oblique_axis,
            r_inner=0.14, r_outer=0.25, length=0.48,
            stress_outer=0.70, stress_bore=0.80, n_sides=36,
        )

        unseen_5boss_stl = tmp_path / "unseen_5boss_oblique_yoke.stl"
        export_binary_stl(np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32), file_path=unseen_5boss_stl)

        loaded = load_blind_cad_file(unseen_5boss_stl)
        feat = recognize_features_from_blind_cad(loaded, grid_shape=(20, 16, 16), v_budget=0.25)
        assert len(feat.bores) == 5
        assert len(feat.bosses) == 5
        # Verify at least one bore preserves a non-axis-aligned oblique 3D unit vector
        has_oblique = any(
            0.20 < abs(float(b.axis[1])) < 0.96 and 0.20 < abs(float(b.axis[2])) < 0.96
            for b in feat.bores
        )
        assert has_oblique is True, f"Expected oblique bore axis vector to be preserved, got {[b.axis for b in feat.bores]}"


class TestReferenceFree8IntrinsicLaws:
    """Verify all 8 Reference-Free Intrinsic Physical, Geometric & Topological Laws across all 5 blind CAD parts."""

    def test_all_5_blind_parts_pass_8_of_8_intrinsic_gates(self, corpus_results):
        """Every blind CAD part in the 5-part corpus must pass 8/8 reference-free intrinsic gates."""
        assert len(corpus_results) == 5
        for res in corpus_results:
            audit = res.intrinsic_audit
            assert audit["reference_free"] is True
            assert audit["gates_passed"] == 8, f"{res.part_id} failed intrinsic gates: {audit['laws']}"
            assert audit["all_passed"] is True

    def test_law_1_stage1_self_baseline_stiffness_efficiency(self, corpus_results):
        """Law 1: Stage 1 self-baseline stiffness efficiency eta_stiff > 1.35 on all 5 parts."""
        for res in corpus_results:
            law1 = res.intrinsic_audit["laws"]["1_stage1_self_baseline_stiffness_efficiency"]
            assert law1["passed"] is True
            assert law1["eta_stiff"] > 1.35
            assert law1["eta_simp_uniform"] > 5.0

    def test_law_2_bore_clearance_survival_100_percent(self, corpus_results):
        """Law 2: 100.0% bore clearance survival (0 intruding voxels or triangles across all 4 stages)."""
        for res in corpus_results:
            law2 = res.intrinsic_audit["laws"]["2_bore_clearance_survival_pct"]
            assert law2["passed"] is True
            assert law2["bore_clearance_survival_pct"] == 100.0
            assert law2["intruding_voxels_total"] == 0
            assert law2["intruding_triangles_total"] == 0
            for st_key in ("stage_1", "stage_2", "stage_3", "stage_4"):
                assert law2["intruding_voxels_by_stage"][st_key] == 0
                assert law2["intruding_triangles_by_stage"][st_key] == 0

    def test_law_3_boss_datum_retention_99_percent(self, corpus_results):
        """Law 3: Mounting boss collar retention >= 99.0% and 0.000 datum erosion."""
        for res in corpus_results:
            law3 = res.intrinsic_audit["laws"]["3_boss_datum_retention_pct"]
            assert law3["passed"] is True
            assert law3["boss_datum_retention_pct"] >= 99.0
            assert law3["datum_erosion"] == 0.0

    def test_law_4_load_path_beta0_connected(self, corpus_results):
        """Law 4: 26-connected solid flood fill verifies beta_0 == 1, 0 floating islands, and load-to-support connectivity."""
        for res in corpus_results:
            law4 = res.intrinsic_audit["laws"]["4_load_path_beta0_connected"]
            assert law4["passed"] is True
            assert law4["beta_0"] == 1
            assert law4["floating_islands"] == 0
            assert law4["load_to_support_connected"] is True

    def test_law_5_true_equilibrium_residual_admissible(self, corpus_results):
        """Law 5: Recomputed true equilibrium residual r_eq <= 5.0e-2 (arXiv:2607.26382v1)."""
        for res in corpus_results:
            law5 = res.intrinsic_audit["laws"]["5_true_equilibrium_residual_admissible"]
            assert law5["passed"] is True
            assert law5["r_eq"] <= 5.0e-2

    def test_law_6_strict_4stage_volume_monotonicity(self, corpus_results):
        """Law 6: Strict 4-stage volume monotonicity V_S1 > V_S2 > V_S3 > V_S4 and |V_S4/V_S1 - V_target| <= 0.015."""
        for res in corpus_results:
            law6 = res.intrinsic_audit["laws"]["6_strict_4stage_volume_monotonicity"]
            assert law6["passed"] is True
            assert law6["is_monotonic"] is True
            vols = law6["stage_volume_fractions"]
            div_vols = law6["stage_divergence_volumes"]
            assert vols[0] > vols[1] > vols[2] > vols[3]
            assert div_vols[0] > div_vols[1] > div_vols[2] > div_vols[3]
            assert law6["stage4_volume_error"] <= 0.015

    def test_law_7_intrinsic_voxel_quality(self, corpus_results):
        """Law 7: 0 trapped beta_2 cavities, 0.0% FEA checkerboards, and >= 98.0% binary crispness."""
        for res in corpus_results:
            law7 = res.intrinsic_audit["laws"]["7_intrinsic_voxel_quality"]
            assert law7["passed"] is True
            assert law7["trapped_beta2_cavities"] == 0
            assert law7["checkerboard_pct"] == 0.0
            assert law7["crispness_pct"] >= 98.0

    def test_law_8_watertight_2manifold_stl_closure_and_genus(self, corpus_results):
        """
        Law 8: Exported binary .STL meshes for all 4 stages satisfy boundary_edges == 0,
        non_manifold_edges == 0, degenerate_faces == 0, min_dot > 0.05, divergence_volume > 0,
        and beta_1(S4) >= beta_1(S1).
        """
        for res in corpus_results:
            law8 = res.intrinsic_audit["laws"]["8_watertight_2manifold_stl_closure"]
            assert law8["passed"] is True
            assert law8["all_stages_2manifold"] is True
            assert law8["beta1_stage4"] >= law8["beta1_stage1"]
            for st_key, st_aud in law8["stages"].items():
                assert st_aud["boundary_edges"] == 0, f"{res.part_id} {st_key} has boundary edges"
                assert st_aud["non_manifold_edges"] == 0, f"{res.part_id} {st_key} has non-manifold edges"
                assert st_aud["degenerate_faces"] == 0, f"{res.part_id} {st_key} has degenerate faces"
                assert st_aud["min_dot"] > 0.05, f"{res.part_id} {st_key} min_dot <= 0.05"
                assert st_aud["divergence_volume"] > 0.0
                assert st_aud["stl_binary_valid"] is True
                # Verify exact Euler-Poincare multi-component Betti formula: b1 == 1 - chi_solid + b2
                chi_solid = (st_aud["V"] - st_aud["E"] + st_aud["F"]) // 2
                b2_expected = max(0, st_aud["connected_components"] - 1)
                b1_expected = max(0, 1 - chi_solid + b2_expected)
                assert st_aud["b1"] == b1_expected

    def test_distinct_3d_fea_telemetry_and_exact_betti_numbers_and_vertex_stress(self, corpus_results):
        """
        Verify that:
        1. Every part has a genuinely distinct 3D FEA stiffness efficiency `eta_stiff` and binary
           crispness `crispness_pct` (never constant/formulaic across the 5 parts).
        2. Stage 1 Betti number `beta1_stage1` equals the number of recognized through-bores
           (3, 3, 3, 3, 4) with zero hidden buried genus cores.
        3. Every stage mesh has non-constant 3D FEA von Mises vertex colors across its surface.
        """
        eta_vals = [
            round(float(res.intrinsic_audit["laws"]["1_stage1_self_baseline_stiffness_efficiency"]["eta_stiff"]), 3)
            for res in corpus_results
        ]
        crisp_vals = [
            round(float(res.intrinsic_audit["laws"]["7_intrinsic_voxel_quality"]["crispness_pct"]), 2)
            for res in corpus_results
        ]
        assert len(set(eta_vals)) == 5, f"Expected 5 distinct eta_stiff values across 5 parts, got {eta_vals}"
        assert len(set(crisp_vals)) >= 4, f"Expected distinct crispness_pct values across parts, got {crisp_vals}"
        for eta_v in eta_vals:
            assert abs(eta_v - 4.87) > 0.10, "eta_stiff must come from true 3D FEA, not formulaic 4.87x"

        for res in corpus_results:
            law8 = res.intrinsic_audit["laws"]["8_watertight_2manifold_stl_closure"]
            assert law8["beta1_stage1"] == len(res.features.bores), (
                f"{res.part_id}: Stage 1 b1 ({law8['beta1_stage1']}) must equal bore count ({len(res.features.bores)})"
            )
            assert law8["beta1_stage4"] > law8["beta1_stage1"], (
                f"{res.part_id}: Stage 4 b1 ({law8['beta1_stage4']}) must strictly exceed Stage 1 b1 ({law8['beta1_stage1']})"
            )
            for st in res.stages:
                c_std = float(np.std(st.vertex_colors))
                assert c_std > 0.05, f"{res.part_id} {st.stage_id}: vertex colors must vary spatially (std={c_std:.4f})"

    def test_fea_salience_field_drives_stage4_arch_and_root_fillet_geometry(self, corpus_results):
        """
        Verify that the 3D FEA Beltrami-Morse strain-energy `salience_field` passed into
        `_extract_progressive_blind_cad_stage_mesh` genuinely modulates Stage 4 arch trajectories
        and root-fillet flares.
        """
        res0 = corpus_results[0]
        Z, Y, X = res0.features.grid_shape
        zz, yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, Z),
            np.linspace(0.0, 1.0, Y),
            np.linspace(0.0, 1.0, X),
            indexing="ij",
        )
        sal_a = (0.1 + 0.9 * (xx ** 2 + zz ** 2)).astype(np.float32)
        sal_b = (0.1 + 0.9 * ((1.0 - xx) ** 2 + (1.0 - zz) ** 2)).astype(np.float32)
        stress_dummy = np.ones((Z, Y, X), dtype=np.float32)

        v_a, f_a, _, _, _ = _extract_progressive_blind_cad_stage_mesh(
            mesh=res0.loaded_mesh,
            features=res0.features,
            st_num=4,
            stress_st=stress_dummy,
            taper_x=0.34,
            waist_y=0.32,
            fen_rings=2,
            salience_field=sal_a,
        )
        v_b, f_b, _, _, _ = _extract_progressive_blind_cad_stage_mesh(
            mesh=res0.loaded_mesh,
            features=res0.features,
            st_num=4,
            stress_st=stress_dummy,
            taper_x=0.34,
            waist_y=0.32,
            fen_rings=2,
            salience_field=sal_b,
        )
        assert v_a.shape == v_b.shape
        assert np.array_equal(f_a, f_b)
        max_vertex_shift = float(np.max(np.abs(v_a - v_b)))
        assert max_vertex_shift > 1e-3, f"Expected 3D FEA salience_field to shift Stage 4 arch geometry, got {max_vertex_shift}"


class TestFourSolverBlindCADContrast:
    """Verify 4-solver blind CAD pathology separation on identical voxelized stock & BCs."""

    def test_4solver_blind_comparison_pathology_separation(self, corpus_results):
        """
        Verify on blind CAD stock that:
        - `UnfilteredSIMP` fails the intrinsic checkerboard gate (checkerboard_pct > 12%),
        - `GaussianFilteredSIMP` fails the intrinsic binary crispness gate (crispness_pct < 78%),
        - `HeavisideProjectedSIMP` exhibits inferior stiffness efficiency and trapped beta_2 >= 1 cavities,
        - `TopoSnail` passes 8/8 intrinsic reference-free gates.
        """
        for res in corpus_results:
            comp = res.solver_comparison
            assert set(comp.keys()) == {"TopoSnail", "UnfilteredSIMP", "GaussianFilteredSIMP", "HeavisideProjectedSIMP"}

            # 1. UnfilteredSIMP fails checkerboard gate (> 12%)
            assert comp["UnfilteredSIMP"]["checkerboard_pct"] > 12.0
            assert comp["UnfilteredSIMP"]["all_passed"] is False

            # 2. GaussianFilteredSIMP fails binary crispness gate (< 78%)
            assert comp["GaussianFilteredSIMP"]["crispness_pct"] < 78.0
            assert comp["GaussianFilteredSIMP"]["all_passed"] is False

            # 3. HeavisideProjectedSIMP leaves trapped beta_2 cavities and lower stiffness efficiency
            assert comp["HeavisideProjectedSIMP"]["trapped_beta2_cavities"] >= 1
            assert comp["HeavisideProjectedSIMP"]["eta_stiff"] < comp["TopoSnail"]["eta_stiff"]
            assert comp["HeavisideProjectedSIMP"]["all_passed"] is False

            # 4. TopoSnail passes 8/8 reference-free gates
            assert comp["TopoSnail"]["gates_passed"] == 8
            assert comp["TopoSnail"]["all_passed"] is True
            assert comp["TopoSnail"]["checkerboard_pct"] == 0.0
            assert comp["TopoSnail"]["crispness_pct"] >= 98.0
            assert comp["TopoSnail"]["trapped_beta2_cavities"] == 0


class TestCLIRunnerAndArtifactGeneration:
    """Verify CLI runner (`scripts/run_blind_cad_benchmark.py`), JSON summary, and PNG visual sheet."""

    def test_cli_single_custom_part_and_artifacts(self, blind_corpus_manifest, tmp_path: Path):
        """Running `scripts/run_blind_cad_benchmark.py --input <custom.stl>` must succeed and emit JSON & PNG."""
        custom_stl = Path(blind_corpus_manifest[0]["stl_path"])
        json_out = tmp_path / "custom_summary.json"
        png_out = tmp_path / "custom_audit.png"

        exit_code = cli_main([
            "--input", str(custom_stl),
            "--summary-json", str(json_out),
            "--audit-png", str(png_out),
            "--resolution", "16,12,12",
            "--max-iters", "10",
        ])
        assert exit_code == 0
        assert json_out.exists() and json_out.stat().st_size > 200
        assert png_out.exists() and png_out.stat().st_size > 10000

        payload = json.loads(json_out.read_text(encoding="utf-8"))
        assert payload["reference_free"] is True
        assert payload["zero_example_model_used"] is True
        assert payload["num_parts_evaluated"] == 1
        assert payload["toposnail_parts_passed_8_of_8"] == 1
        assert payload["all_passed"] is True

        table_str = format_blind_cad_benchmark_tables(payload)
        assert "TOPOSNAIL REFERENCE-FREE BLIND CAD BENCHMARK" in table_str
        assert "TopoSnail" in table_str
        assert "UnfilteredSIMP" in table_str
        assert "GaussianFilteredSIMP" in table_str
        assert "HeavisideProjectedSIMP" in table_str
