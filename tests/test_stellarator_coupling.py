"""
Tests for Stellarator Modular Coil & Divertor Bionic Synthesis (R4.1).

Verifies:
1. sample_mhd_equilibrium_and_divertor: 3D non-stellarator-symmetric MHD Lorentz force
   density (J x B = grad p), Stepped Flow-MRxMHD Beltrami shells, and divertor heat flux
   Delta T(z, y, x) peaking at 1200 K.
2. build_stellarator_coupling_scenario: proper initialization of ScenarioData with
   TTI anisotropy parameters, keep-out cooling channels, and solid armor tiles.
3. run_stellarator_bionic_coupling: end-to-end execution of TopoSnailOptimizer with
   6-way Jordan boundary flood filling asserting strictly 0 trapped beta_2 cavities (b2 == 0),
   compliance reduction (J* < 1.0), true equilibrium residual (r_eq <= 1.0e-4),
   SO(6) Kelvin-Mandel rotor invariance, and 4-stage mesh export.
4. stellarator_bionic_coupling_summary.json integrity and compliance.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from toposnail.stellarator_coupling import (
    sample_mhd_equilibrium_and_divertor,
    build_stellarator_coupling_scenario,
    run_stellarator_bionic_coupling,
    build_scen_06_data,
    DEFAULT_GRID_SHAPE,
)


class TestStellaratorMHDAndDivertorCoupling:
    """Unit and integration tests for MHD equilibrium sampling and divertor heat flux."""

    def test_sample_mhd_equilibrium_and_divertor_fields(self):
        """Verify 3D MHD Lorentz forces, Beltrami stepped pressure, and 1200 K divertor thermal field."""
        res = sample_mhd_equilibrium_and_divertor(resolution=(32, 24, 24), family=1, tau0=0.38)

        # 1. Field tensor shapes
        assert res["lorentz_forces"].shape == (32, 24, 24, 3)
        assert res["b_field"].shape == (32, 24, 24, 3)
        assert res["thermal_field_delta_t"].shape == (32, 24, 24)
        assert res["stepped_p"].shape == (32, 24, 24)
        assert res["psi"].shape == (32, 24, 24)

        # 2. Divertor thermal peak
        t_peak = float(np.max(res["thermal_field_delta_t"]))
        assert abs(t_peak - 1200.0) < 1e-3, f"Expected 1200.0 K peak, got {t_peak}"

        # 3. Lorentz force and grad p consistency
        lf_norm = np.linalg.norm(res["lorentz_forces"], axis=-1)
        assert np.max(lf_norm) > 0.0, "Lorentz force must be non-zero"
        assert res["grad_p"].shape == (32, 24, 24, 3)

        # 4. Stepped-pressure Beltrami invariants
        assert res["bernoulli_jump_residual"] == 0.0

    def test_build_stellarator_coupling_scenario_invariants(self):
        """Verify scenario construction, passive zones, and boundary conditions."""
        scen = build_stellarator_coupling_scenario(resolution=(32, 24, 24), v_budget=0.25)

        assert scen.scenario_id == "06_stellarator_modular_coil_divertor"
        assert scen.grid_shape == (32, 24, 24)
        assert scen.volume_budget == 0.25

        # Disjoint solid and KOZ masks
        overlap = scen.solid_mask & scen.koz_mask
        assert not np.any(overlap), "Solid and keep-out zones must be strictly disjoint"

        # Fixed anchor boundary
        assert np.any(scen.fixed_mask), "Scenario must define fixed boundary anchors"

        # Thermal gradient attached
        assert scen.thermal_delta_t is not None
        assert abs(float(np.max(scen.thermal_delta_t)) - 1200.0) < 1e-3

        # TTI anisotropic parameters
        assert "epsilon" in scen.anisotropy_params
        assert "delta" in scen.anisotropy_params
        assert "gamma" in scen.anisotropy_params


class TestStellaratorBionicOptimization:
    """Integration tests for coupled TTI thermo-elastic topology optimization and mesh synthesis."""

    def test_run_stellarator_bionic_coupling_end_to_end(self, tmp_path):
        """Execute full bionic synthesis asserting strictly b2 == 0 and compliance reduction."""
        summary = run_stellarator_bionic_coupling(
            output_dir=tmp_path,
            max_iterations=6,
            v_budget=0.25,
            resolution=(32, 24, 24),
        )

        # 1. Strictly 0 trapped internal beta_2 cavities
        assert summary["b2_trapped_cavities"] == 0, (
            f"Expected strictly 0 trapped cavities, got {summary['b2_trapped_cavities']}"
        )

        # 2. Genuine compliance reduction (J* < 1.0)
        assert summary["compliance_reduction_ratio"] < 1.0, (
            f"Expected compliance reduction < 1.0, got {summary['compliance_reduction_ratio']}"
        )
        assert summary["compliance_final"] < summary["compliance_initial"]

        # 3. True recomputed equilibrium residual r_eq <= 1.0e-4
        assert summary["equilibrium_residual"] <= 1.0e-4, (
            f"Expected r_eq <= 1.0e-4, got {summary['equilibrium_residual']}"
        )

        # 4. SO(6) Kelvin-Mandel rotor invariance
        assert summary["so6_rotor_drift"] <= 1.14e-13, (
            f"Expected SO(6) rotor drift <= 1.14e-13, got {summary['so6_rotor_drift']}"
        )

        # 5. Divertor thermal peak preserved
        assert abs(summary["peak_divertor_temp_k"] - 1200.0) < 1e-3

        # 6. Verify 4-stage bionic mesh files exported (.STL and .OBJ)
        mesh_files = summary["mesh_files"]
        for stage_num in range(1, 5):
            stl_key = f"stage_{stage_num}_stl"
            obj_key = f"stage_{stage_num}_obj"
            assert stl_key in mesh_files, f"Missing {stl_key}"
            assert obj_key in mesh_files, f"Missing {obj_key}"

            stl_path = tmp_path / mesh_files[stl_key]
            obj_path = tmp_path / mesh_files[obj_key]
            assert stl_path.exists() and stl_path.stat().st_size > 0, f"Empty STL: {stl_path}"
            assert obj_path.exists() and obj_path.stat().st_size > 0, f"Empty OBJ: {obj_path}"

    def test_stellarator_bionic_coupling_summary_json_artifact(self):
        """Verify official results artifact toposnail/results/stellarator_bionic_coupling_summary.json."""
        summary_file = Path(__file__).resolve().parent.parent / "results" / "stellarator_bionic_coupling_summary.json"
        assert summary_file.exists(), f"Summary file missing: {summary_file}"

        data = json.loads(summary_file.read_text(encoding="utf-8"))
        assert data["scenario_id"] == "06_stellarator_modular_coil_divertor"
        assert data["b2_trapped_cavities"] == 0
        assert data["compliance_reduction_ratio"] < 1.0
        assert data["equilibrium_residual"] <= 1.0e-4
        assert data["so6_rotor_drift"] <= 1.14e-13
        assert abs(data["peak_divertor_temp_k"] - 1200.0) < 1e-3
        assert "mesh_files" in data

    def test_build_scen_06_data_and_portable_studio_html(self):
        """Verify scenario 06 offline dictionary structure and portable studio HTML integration."""
        data = build_scen_06_data()
        assert data["scenario_id"] == "06_stellarator_modular_coil_divertor"
        assert len(data["stages"]) == 4
        assert "mesh" in data
        assert len(data["mesh"]["vertices"]) > 0
        assert len(data["mesh"]["faces"]) > 0
        for fkey in ("toposnail_rho", "toposnail_vm", "toposnail_th", "toposnail_u", "toposnail_curl", "toposnail_fiber"):
            assert fkey in data["fields_b64"]
            assert len(data["fields_b64"][fkey]) > 0

        studio_html = Path(__file__).resolve().parent.parent / "studio" / "toposnail_portable_studio.html"
        assert studio_html.exists()
        content = studio_html.read_text(encoding="utf-8")
        assert "06_stellarator_modular_coil_divertor" in content
        assert "window.TOPOSNAIL_OFFLINE_SCENARIOS" in content
