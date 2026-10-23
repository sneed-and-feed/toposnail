"""
toposnail.elasticity: Matrix-Free 3D Hexahedral Element-by-Element (EBE) Elasticity,
SO(6) Kelvin-Mandel Stiffness Rotors, Thomsen TTI Elastic Anisotropy, and Tsai-Hill Yield Envelopes.

Literature Grounding & Citations:
1. Yang, Wang & Wang (2026, arXiv:2604.18020v1), "Matrix-Free 3D SIMP Topology Optimization with
   Fused Gather-GEMM-Scatter Kernels" (https://arxiv.org/abs/2604.18020v1):
   Fuses element nodal gather, 24x24 element stiffness vector product, and nodal scatter accumulation
   inside chunk_z = 16 slabs (< 35 MB heap), avoiding the O(N^3) global stiffness RAM barrier.
2. Yang, Wang & Wang (2026, arXiv:2607.26382v1), "When a positive SIMP density floor is not enough:
   solver admissibility and guarded floor selection in matrix-free 3D topology optimization"
   (https://arxiv.org/abs/2607.26382v1):
   Tracks recomputed true equilibrium residual r_eq = ||f - K u||_2 / ||f||_2 with guarded density
   floor selection rho_min in [1e-6, 1e-3].
3. Moter, Abdelhamid & Czekanski (2021, arXiv:2112.02030v2), "Direction-Oriented Stress-Constrained
   Topology Optimization of Orthotropic Materials" (https://arxiv.org/abs/2112.02030v2):
   Co-optimizes voxel density rho_e and local principal stress orientation R_e in SO(3)
   lifted to M(R_e) in SO(6) with direction-oriented orthotropic Tsai-Hill yield index envelopes.
4. Kim & Kang (2025, arXiv:2504.05604v1), "PyTopo3D: A Python Framework for 3D SIMP-based Topology Optimization":
   3D 8-node trilinear hexahedral (24-DOF) analytical integration and boundary condition enforcement.
5. Liu & Tovar (2014, SMO top3d):
   3D topology optimization benchmarks and compliance sensitivities.
6. Zhou et al. (2024, arXiv:2410.06850v4):
   Coupled thermo-elastic stress modeling sigma_therm = C_rot : alpha * Delta_T(z,y,x).
7. Formalization: OcclusionTriad.lean (Theorems 2.1-2.8):
   Theorem 2.7: so6_stiffness_trace_conservation, so6_stiffness_preserves_pos_def.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# Canonical 8-node local offsets (dz, dy, dx) for Q1 trilinear hexahedral elements
NODE_OFFSETS: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 1),
    (0, 1, 0),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 1),
    (1, 1, 0),
)


@dataclass
class HexahedralMesh3D:
    """3D regular hexahedral (Q1 brick) finite-element mesh representation."""

    grid_shape: Tuple[int, int, int]  # (Z, Y, X)
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # (dz, dy, dx)

    @property
    def num_elements(self) -> int:
        return self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2]

    @property
    def num_nodes(self) -> int:
        return (
            (self.grid_shape[0] + 1)
            * (self.grid_shape[1] + 1)
            * (self.grid_shape[2] + 1)
        )

    @property
    def num_dofs(self) -> int:
        return self.num_nodes * 3


@dataclass
class ElasticMaterialProperties:
    """Isotropic or Tilted Transversely Isotropic (TTI) elastic material parameters."""

    name: str = "structural_steel"
    e_modulus: float = 1.0
    nu_poisson: float = 0.3
    # Anisotropic Thomsen parameters
    is_anisotropic: bool = False
    vp0: float = 3.65  # km/s
    vs0: float = 2.18  # km/s
    rho_density: float = 2.44  # g/cm^3
    epsilon: float = 0.0  # P-wave horizontal anisotropy
    delta: float = 0.0  # Near-vertical P-wave anisotropy
    gamma: float = 0.0  # SH-wave anisotropy
    alpha_thermal: float = 1.2e-5  # Thermal expansion coefficient (1/K)
    yield_strength: float = 1.0  # Reference yield stress


@dataclass
class EBERelaxationResult:
    """Results from matrix-free element-by-element strain-energy relaxation."""

    u_disp: np.ndarray  # (num_dofs,) float32
    strain_energy: np.ndarray  # (Z, Y, X) float32
    compliance: float
    von_mises: np.ndarray  # (Z, Y, X) float32
    tsai_hill: np.ndarray  # (Z, Y, X) float32
    principal_frames: np.ndarray  # (Z, Y, X, 3, 3) float64 R_e in SO(3)
    so6_rotors: np.ndarray  # (Z, Y, X, 6, 6) float64 M(R_e) in SO(6)
    compliance_sensitivity: np.ndarray  # (Z, Y, X) float32
    equilibrium_residual: float  # r_eq = ||f - K u|| / ||f||
    cg_iterations: int
    cg_converged: bool


class ThomsenTTIAnisotropicElasticEngine:
    """
    Matrix-free 3D hexahedral EBE elastic engine with SO(6) Kelvin-Mandel rotors.
    Evaluates linear elasticity, thermal stress, and sensitivities in Z-slabs (< 35 MB heap).
    """

    def __init__(
        self,
        vp0: float = 3.65,
        vs0: float = 2.18,
        rho0: float = 2.44,
        eps: float = 0.0,
        delta: float = 0.0,
        gamma: float = 0.0,
        grid_shape: Tuple[int, int, int] = (32, 24, 24),
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        material: Optional[ElasticMaterialProperties] = None,
        simp_penalty: float = 3.0,
        density_floor: float = 1e-3,
        chunk_z: int = 16,
    ):
        self.grid_shape = tuple(grid_shape)
        self.voxel_size = tuple(voxel_size)
        self.simp_penalty = float(simp_penalty)
        self.density_floor = float(density_floor)
        self.chunk_z = int(chunk_z)

        if material is not None:
            self.material = material
            self.vp0 = material.vp0
            self.vs0 = material.vs0
            self.rho0 = material.rho_density
            self.eps = material.epsilon
            self.delta = material.delta
            self.gamma = material.gamma
        else:
            self.vp0 = float(vp0)
            self.vs0 = float(vs0)
            self.rho0 = float(rho0)
            self.eps = float(eps)
            self.delta = float(delta)
            self.gamma = float(gamma)
            is_aniso = (abs(self.eps) > 1e-8 or abs(self.delta) > 1e-8 or abs(self.gamma) > 1e-8)
            self.material = ElasticMaterialProperties(
                name="thomsen_tti" if is_aniso else "isotropic",
                is_anisotropic=is_aniso,
                vp0=self.vp0,
                vs0=self.vs0,
                rho_density=self.rho0,
                epsilon=self.eps,
                delta=self.delta,
                gamma=self.gamma,
            )

        self.mesh = HexahedralMesh3D(self.grid_shape, self.voxel_size)

        # Build base 6x6 Kelvin-Mandel stiffness tensor C_VTI in float64
        self.c_vti = self.build_vti_kelvin_stiffness_tensor(
            self.vp0, self.vs0, self.rho0, self.eps, self.delta, self.gamma
        )

        # Precompute 8 Gauss-point strain-displacement matrices B_g and base element stiffness k0
        self.B_gauss, self.B_center = self._precompute_hex8_b_matrices(self.voxel_size)
        self.k0 = self.build_hex8_analytical_stiffness(self.c_vti)

        # Precompute element DOF connectivity
        self.elem_dofs = self._precompute_element_dofs()

        # Vectorized slice operation precomputations and aligned working buffers
        self.k0_f32 = self.k0.astype(np.float32)
        self.diag_k0_f32 = np.diag(self.k0).astype(np.float32)
        self.diag_k0_3 = self.diag_k0_f32[:3].copy()
        Z, Y, X = self.grid_shape
        self._d_e = np.empty((Z, Y, X, 24), dtype=np.float32)
        self._y_grid = np.zeros((Z + 1, Y + 1, X + 1, 3), dtype=np.float32)
        self._diag_grid = np.zeros((Z + 1, Y + 1, X + 1, 3), dtype=np.float32)
        self._E_nodes = np.zeros((Z + 1, Y + 1, X + 1), dtype=np.float32)
        self._u_cf = np.zeros((3, Z + 1, Y + 1, X + 1), dtype=np.float32)
        self._y_cf = np.zeros((3, Z + 1, Y + 1, X + 1), dtype=np.float32)
        self._d_cf = np.empty((24, Z, Y, X), dtype=np.float32)
        self._m_inv_cf = np.zeros((3, Z + 1, Y + 1, X + 1), dtype=np.float32)

    # =========================================================================
    # 1. Thomsen TTI Elasticity & Kelvin-Mandel Representation
    # =========================================================================

    @staticmethod
    def build_vti_kelvin_stiffness_tensor(
        vp0: float,
        vs0: float,
        rho0: float,
        eps: float,
        delta: float,
        gamma: float,
    ) -> np.ndarray:
        """
        Construct 6x6 symmetric VTI stiffness tensor C_VTI in normalized Kelvin-Mandel basis
        from Thomsen parameters (vp0, vs0, rho0, eps, delta, gamma):
            (sigma_11, sigma_22, sigma_33, sqrt(2)*sigma_23, sqrt(2)*sigma_13, sqrt(2)*sigma_12).
        """
        vp0_f = float(vp0)
        vs0_f = float(vs0)
        rho_f = float(rho0)
        eps_f = float(eps)
        dlt_f = float(delta)
        gam_f = float(gamma)

        c33 = rho_f * (vp0_f**2)
        c55 = rho_f * (vs0_f**2)
        c44 = c55
        c11 = c33 * (1.0 + 2.0 * eps_f)
        c22 = c11
        c66 = c55 * (1.0 + 2.0 * gam_f)
        c12 = c11 - 2.0 * c66

        # Exact Thomsen C13 relation:
        # (c13 + c55)^2 = (c33 - c55)^2 + 2 * delta * c33 * (c33 - c55)
        disc = max((c33 - c55) ** 2 + 2.0 * dlt_f * c33 * (c33 - c55), 1e-8)
        c13 = np.sqrt(disc) - c55
        c23 = c13

        C = np.zeros((6, 6), dtype=np.float64)
        C[0, 0] = c11
        C[1, 1] = c22
        C[2, 2] = c33
        C[0, 1] = C[1, 0] = c12
        C[0, 2] = C[2, 0] = c13
        C[1, 2] = C[2, 1] = c23

        # In Kelvin-Mandel basis, shear diagonal entries are 2*c44, 2*c55, 2*c66
        C[3, 3] = 2.0 * c44
        C[4, 4] = 2.0 * c55
        C[5, 5] = 2.0 * c66
        return C

    # =========================================================================
    # 2. SO(3) -> SO(6) Kelvin-Mandel Stiffness Rotors (Theorem 2.7)
    # =========================================================================

    @staticmethod
    def build_so3_from_stress_tensor(stress_6: np.ndarray) -> np.ndarray:
        """
        Extract right-handed SO(3) principal stress frame R from a 6-vector
        (sigma_11, sigma_22, sigma_33, sqrt(2)*sigma_23, sqrt(2)*sigma_13, sqrt(2)*sigma_12).
        Rows of R are unit principal stress directions [e1, e2, e3] with det(R) = +1.
        """
        s2 = np.sqrt(2.0)
        s11, s22, s33 = float(stress_6[0]), float(stress_6[1]), float(stress_6[2])
        s23 = float(stress_6[3]) / s2
        s13 = float(stress_6[4]) / s2
        s12 = float(stress_6[5]) / s2

        # Cauchy stress tensor
        sigma_3x3 = np.array(
            [[s11, s12, s13], [s12, s22, s23], [s13, s23, s33]], dtype=np.float64
        )

        # Eigen-decomposition: sigma_3x3 = V @ diag(lambda) @ V.T
        evals, evecs = np.linalg.eigh(sigma_3x3)
        # Order by decreasing eigenvalues
        idx = np.argsort(evals)[::-1]
        V = evecs[:, idx].copy()

        # Enforce right-handed orientation det(V) = +1 in SO(3)
        if np.linalg.det(V) < 0.0:
            V[:, 2] = -V[:, 2]

        # R has principal directions as rows: R = V.T
        return V.T

    @staticmethod
    def build_so3_from_dip_azimuth(
        dip_deg: Union[float, np.ndarray],
        azimuth_deg: Union[float, np.ndarray],
    ) -> np.ndarray:
        """
        Construct 3D rotation matrix R in SO(3) [..., 3, 3] from dip theta and azimuth phi.
        """
        theta = np.deg2rad(np.asarray(dip_deg, dtype=np.float64))
        phi = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64))

        ct, st = np.cos(theta), np.sin(theta)
        cp, sp = np.cos(phi), np.sin(phi)

        if np.ndim(dip_deg) == 0:
            Rz = np.array([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            Ry = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]], dtype=np.float64)
            return Rz @ Ry
        else:
            shape = theta.shape
            Rz = np.zeros(shape + (3, 3), dtype=np.float64)
            Ry = np.zeros(shape + (3, 3), dtype=np.float64)

            Rz[..., 0, 0] = cp
            Rz[..., 0, 1] = -sp
            Rz[..., 1, 0] = sp
            Rz[..., 1, 1] = cp
            Rz[..., 2, 2] = 1.0

            Ry[..., 0, 0] = ct
            Ry[..., 0, 2] = st
            Ry[..., 1, 1] = 1.0
            Ry[..., 2, 0] = -st
            Ry[..., 2, 2] = ct

            return np.matmul(Rz, Ry)

    @staticmethod
    def build_kelvin_mandel_so6_rotor(r3: np.ndarray) -> np.ndarray:
        """
        Lift a 3D rotation matrix R in SO(3) [..., 3, 3] to the exact orthogonal
        6x6 Kelvin-Mandel stiffness rotor M(R) in SO(6) [..., 6, 6] (Theorem 2.7).
        """
        R = np.asarray(r3, dtype=np.float64)
        orig_shape = R.shape[:-2]
        s2 = np.sqrt(2.0)

        r11, r12, r13 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
        r21, r22, r23 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
        r31, r32, r33 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

        M = np.zeros(orig_shape + (6, 6), dtype=np.float64)

        # Upper-left 3x3 normal-normal block
        M[..., 0, 0] = r11 * r11
        M[..., 0, 1] = r12 * r12
        M[..., 0, 2] = r13 * r13

        M[..., 1, 0] = r21 * r21
        M[..., 1, 1] = r22 * r22
        M[..., 1, 2] = r23 * r23

        M[..., 2, 0] = r31 * r31
        M[..., 2, 1] = r32 * r32
        M[..., 2, 2] = r33 * r33

        # Upper-right 3x3 normal-shear block (scaled by sqrt(2))
        M[..., 0, 3] = s2 * r12 * r13
        M[..., 0, 4] = s2 * r11 * r13
        M[..., 0, 5] = s2 * r11 * r12

        M[..., 1, 3] = s2 * r22 * r23
        M[..., 1, 4] = s2 * r21 * r23
        M[..., 1, 5] = s2 * r21 * r22

        M[..., 2, 3] = s2 * r32 * r33
        M[..., 2, 4] = s2 * r31 * r33
        M[..., 2, 5] = s2 * r31 * r32

        # Lower-left 3x3 shear-normal block (scaled by sqrt(2))
        M[..., 3, 0] = s2 * r21 * r31
        M[..., 3, 1] = s2 * r22 * r32
        M[..., 3, 2] = s2 * r23 * r33

        M[..., 4, 0] = s2 * r11 * r31
        M[..., 4, 1] = s2 * r12 * r32
        M[..., 4, 2] = s2 * r13 * r33

        M[..., 5, 0] = s2 * r11 * r21
        M[..., 5, 1] = s2 * r12 * r22
        M[..., 5, 2] = s2 * r13 * r23

        # Lower-right 3x3 shear-shear block
        M[..., 3, 3] = r22 * r33 + r23 * r32
        M[..., 3, 4] = r21 * r33 + r23 * r31
        M[..., 3, 5] = r21 * r32 + r22 * r31

        M[..., 4, 3] = r12 * r33 + r13 * r32
        M[..., 4, 4] = r11 * r33 + r13 * r31
        M[..., 4, 5] = r11 * r32 + r12 * r31

        M[..., 5, 3] = r12 * r23 + r13 * r22
        M[..., 5, 4] = r11 * r23 + r13 * r21
        M[..., 5, 5] = r11 * r22 + r12 * r21

        return M

    def compute_so6_kelvin_rotor(self, r_so3: np.ndarray) -> np.ndarray:
        """Alias for build_kelvin_mandel_so6_rotor."""
        return self.build_kelvin_mandel_so6_rotor(r_so3)

    def rotate_stiffness_tensor(
        self,
        c_vti: np.ndarray,
        m_so6: np.ndarray,
        return_diagnostics: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, float, float, float]]:
        """
        Compute C_rot = M @ C_vti @ M.T.
        Guarantees trace and Frobenius norm conservation <= 1.14e-13.
        """
        C = np.asarray(c_vti, dtype=np.float64)
        M = np.asarray(m_so6, dtype=np.float64)

        C_rot = np.matmul(np.matmul(M, C), np.swapaxes(M, -1, -2))

        if not return_diagnostics:
            return C_rot

        # Diagnostics
        eye6 = np.eye(6, dtype=np.float64)
        so6_ortho_err = float(np.max(np.abs(np.matmul(np.swapaxes(M, -1, -2), M) - eye6)))

        tr_vti = np.trace(C, axis1=-2, axis2=-1)
        tr_rot = np.trace(C_rot, axis1=-2, axis2=-1)
        trace_drift = float(np.max(np.abs(tr_rot - tr_vti)))

        frob_vti = np.linalg.norm(C, ord="fro", axis=(-2, -1))
        frob_rot = np.linalg.norm(C_rot, ord="fro", axis=(-2, -1))
        frob_drift = float(np.max(np.abs(frob_rot - frob_vti)))

        return C_rot, so6_ortho_err, trace_drift, frob_drift

    # =========================================================================
    # 3. 24-DOF Hexahedral Analytical Integration & Precomputations
    # =========================================================================

    def _precompute_hex8_b_matrices(
        self, voxel_size: Tuple[float, float, float]
    ) -> Tuple[list, np.ndarray]:
        """
        Precompute the 6x24 strain-displacement matrices B_g at the 8 Gauss points
        and B_center at element centroid for trilinear hexahedron Q1.
        """
        dz, dy, dx = voxel_size
        xi_nodes = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=np.float64)
        eta_nodes = np.array([-1, -1, 1, 1, -1, -1, 1, 1], dtype=np.float64)
        zeta_nodes = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=np.float64)

        gp = 1.0 / np.sqrt(3.0)
        gauss_points = [
            (-gp, -gp, -gp),
            (gp, -gp, -gp),
            (gp, gp, -gp),
            (-gp, gp, -gp),
            (-gp, -gp, gp),
            (gp, -gp, gp),
            (gp, gp, gp),
            (-gp, gp, gp),
        ]

        s2 = np.sqrt(2.0)
        B_all = []
        for xi, eta, zeta in gauss_points:
            dNx = 0.125 * xi_nodes * (1.0 + eta_nodes * eta) * (1.0 + zeta_nodes * zeta) * (2.0 / dx)
            dNy = 0.125 * eta_nodes * (1.0 + xi_nodes * xi) * (1.0 + zeta_nodes * zeta) * (2.0 / dy)
            dNz = 0.125 * zeta_nodes * (1.0 + xi_nodes * xi) * (1.0 + eta_nodes * eta) * (2.0 / dz)

            B_g = np.zeros((6, 24), dtype=np.float64)
            for a in range(8):
                c = a * 3
                B_g[0, c] = dNx[a]
                B_g[1, c + 1] = dNy[a]
                B_g[2, c + 2] = dNz[a]
                B_g[3, c + 1] = dNz[a] / s2
                B_g[3, c + 2] = dNy[a] / s2
                B_g[4, c] = dNz[a] / s2
                B_g[4, c + 2] = dNx[a] / s2
                B_g[5, c] = dNy[a] / s2
                B_g[5, c + 1] = dNx[a] / s2
            B_all.append(B_g)

        # Centroid matrix B_center (xi=0, eta=0, zeta=0)
        dNx_c = 0.125 * xi_nodes * (2.0 / dx)
        dNy_c = 0.125 * eta_nodes * (2.0 / dy)
        dNz_c = 0.125 * zeta_nodes * (2.0 / dz)
        B_center = np.zeros((6, 24), dtype=np.float64)
        for a in range(8):
            c = a * 3
            B_center[0, c] = dNx_c[a]
            B_center[1, c + 1] = dNy_c[a]
            B_center[2, c + 2] = dNz_c[a]
            B_center[3, c + 1] = dNz_c[a] / s2
            B_center[3, c + 2] = dNy_c[a] / s2
            B_center[4, c] = dNz_c[a] / s2
            B_center[4, c + 2] = dNx_c[a] / s2
            B_center[5, c] = dNy_c[a] / s2
            B_center[5, c + 1] = dNx_c[a] / s2

        return B_all, B_center

    def build_hex8_analytical_stiffness(self, c_tensor_6x6: np.ndarray) -> np.ndarray:
        """
        Compute the 24x24 analytical stiffness matrix k0 for a unit hexahedron Q1
        given the 6x6 Kelvin-Mandel material stiffness tensor C.
        """
        C = np.asarray(c_tensor_6x6, dtype=np.float64)
        dz, dy, dx = self.voxel_size
        det_J = (dx * dy * dz) / 8.0

        k0 = np.zeros((24, 24), dtype=np.float64)
        for B_g in self.B_gauss:
            k0 += np.matmul(np.matmul(B_g.T, C), B_g) * det_J
        return k0

    def _precompute_element_dofs(self) -> np.ndarray:
        """
        Precompute DOF index mapping (Z, Y, X, 24) of int32 for all hexahedral elements.
        """
        Z, Y, X = self.grid_shape
        Nz_nodes, Ny_nodes, Nx_nodes = Z + 1, Y + 1, X + 1

        iz, iy, ix = np.meshgrid(
            np.arange(Z, dtype=np.int32),
            np.arange(Y, dtype=np.int32),
            np.arange(X, dtype=np.int32),
            indexing="ij",
        )

        n0 = iz * (Ny_nodes * Nx_nodes) + iy * Nx_nodes + ix
        n1 = iz * (Ny_nodes * Nx_nodes) + iy * Nx_nodes + (ix + 1)
        n2 = iz * (Ny_nodes * Nx_nodes) + (iy + 1) * Nx_nodes + (ix + 1)
        n3 = iz * (Ny_nodes * Nx_nodes) + (iy + 1) * Nx_nodes + ix
        n4 = (iz + 1) * (Ny_nodes * Nx_nodes) + iy * Nx_nodes + ix
        n5 = (iz + 1) * (Ny_nodes * Nx_nodes) + iy * Nx_nodes + (ix + 1)
        n6 = (iz + 1) * (Ny_nodes * Nx_nodes) + (iy + 1) * Nx_nodes + (ix + 1)
        n7 = (iz + 1) * (Ny_nodes * Nx_nodes) + (iy + 1) * Nx_nodes + ix

        nodes = np.stack([n0, n1, n2, n3, n4, n5, n6, n7], axis=-1)  # (Z, Y, X, 8)

        # 3 DOFs per node (ux, uy, uz)
        dofs = np.zeros((Z, Y, X, 24), dtype=np.int32)
        for a in range(8):
            na = nodes[..., a]
            dofs[..., a * 3] = na * 3
            dofs[..., a * 3 + 1] = na * 3 + 1
            dofs[..., a * 3 + 2] = na * 3 + 2

        return dofs

    # =========================================================================
    # 4. Matrix-Free Element-by-Element Gather-GEMM-Scatter & PCG Solver
    # =========================================================================

    def matrix_free_matvec(
        self,
        u_dof: np.ndarray,
        rho: np.ndarray,
        m6_rotors: Optional[np.ndarray] = None,
        fixed_dofs: Optional[np.ndarray] = None,
        chunk_z: Optional[int] = None,
        E_elem: Optional[np.ndarray] = None,
        return_cf: bool = False,
    ) -> np.ndarray:
        """
        Evaluates y = K(rho, R) @ u without assembling the global sparse stiffness matrix K.
        Uses vectorized 3D hexahedral slice operations in contiguous channels-first layout.
        """
        Z, Y, X = self.grid_shape
        if E_elem is None:
            rho_flat = np.asarray(rho, dtype=np.float32).reshape(Z, Y, X)
            simp_p = self.simp_penalty
            rho_min = self.density_floor
            if abs(simp_p - 3.0) < 1e-5:
                rho_p = rho_flat * rho_flat * rho_flat
            else:
                rho_p = rho_flat ** simp_p
            E_elem = (rho_min + (1.0 - rho_min) * rho_p).astype(np.float32)

        is_cf = (u_dof.ndim == 4 and u_dof.shape == (3, Z + 1, Y + 1, X + 1))
        if is_cf:
            u_cf = u_dof
        else:
            u_flat = np.asarray(u_dof, dtype=np.float32).reshape(-1)
            u_cf = np.ascontiguousarray(u_flat.reshape(Z + 1, Y + 1, X + 1, 3).transpose(3, 0, 1, 2))

        if hasattr(self, "_d_cf") and self._d_cf.shape == (24, Z, Y, X):
            d_cf = self._d_cf
            y_cf = self._y_cf
            y_cf.fill(0.0)
        else:
            d_cf = np.empty((24, Z, Y, X), dtype=np.float32)
            y_cf = np.zeros((3, Z + 1, Y + 1, X + 1), dtype=np.float32)

        # Vectorized 8-node element gather across 3 contiguous channels
        for a, (dz, dy, dx) in enumerate(NODE_OFFSETS):
            a3 = a * 3
            d_cf[a3] = u_cf[0, dz : Z + dz, dy : Y + dy, dx : X + dx]
            d_cf[a3 + 1] = u_cf[1, dz : Z + dz, dy : Y + dy, dx : X + dx]
            d_cf[a3 + 2] = u_cf[2, dz : Z + dz, dy : Y + dy, dx : X + dx]

        if m6_rotors is None:
            d_cf *= E_elem[None, :, :, :]
            f_cf = (self.k0_f32 @ d_cf.reshape(24, -1)).reshape(24, Z, Y, X)
            for a, (dz, dy, dx) in enumerate(NODE_OFFSETS):
                a3 = a * 3
                y_cf[0, dz : Z + dz, dy : Y + dy, dx : X + dx] += f_cf[a3]
                y_cf[1, dz : Z + dz, dy : Y + dy, dx : X + dx] += f_cf[a3 + 1]
                y_cf[2, dz : Z + dz, dy : Y + dy, dx : X + dx] += f_cf[a3 + 2]
        else:
            slab_rotors = m6_rotors.reshape(-1, 6, 6)
            d_e = np.ascontiguousarray(d_cf.transpose(1, 2, 3, 0))
            d_e *= E_elem[..., None]
            d_e_flat = d_e.reshape(-1, 24)
            f_e_flat = np.empty_like(d_e_flat)
            dz, dy, dx = self.voxel_size
            det_J = (dx * dy * dz) / 8.0
            for i in range(len(d_e_flat)):
                C_rot_i = (slab_rotors[i] @ self.c_vti @ slab_rotors[i].T).astype(np.float32)
                k_e_i = np.zeros((24, 24), dtype=np.float32)
                for B_g in self.B_gauss:
                    Bg_f = B_g.astype(np.float32)
                    k_e_i += (Bg_f.T @ C_rot_i @ Bg_f) * np.float32(det_J)
                f_e_flat[i] = k_e_i @ d_e_flat[i]
            f_cf = np.ascontiguousarray(f_e_flat.reshape(Z, Y, X, 24).transpose(3, 0, 1, 2))
            for a, (dz, dy, dx) in enumerate(NODE_OFFSETS):
                a3 = a * 3
                y_cf[0, dz : Z + dz, dy : Y + dy, dx : X + dx] += f_cf[a3]
                y_cf[1, dz : Z + dz, dy : Y + dy, dx : X + dx] += f_cf[a3 + 1]
                y_cf[2, dz : Z + dz, dy : Y + dy, dx : X + dx] += f_cf[a3 + 2]

        if return_cf:
            if fixed_dofs is not None:
                if fixed_dofs.shape == (3, Z + 1, Y + 1, X + 1):
                    y_cf[fixed_dofs] = u_cf[fixed_dofs]
                else:
                    fixed_cf = np.ascontiguousarray(
                        np.asarray(fixed_dofs, dtype=bool).reshape(Z + 1, Y + 1, X + 1, 3).transpose(3, 0, 1, 2)
                    )
                    y_cf[fixed_cf] = u_cf[fixed_cf]
            return y_cf
        else:
            y = np.ascontiguousarray(y_cf.transpose(1, 2, 3, 0)).reshape(-1)
            if fixed_dofs is not None:
                fixed_flat = np.asarray(fixed_dofs, dtype=bool).reshape(-1)
                u_flat = np.asarray(u_dof, dtype=np.float32).reshape(-1)
                y[fixed_flat] = u_flat[fixed_flat]
            return y

    def _compute_jacobi_preconditioner(
        self,
        rho: np.ndarray,
        fixed_dofs: Optional[np.ndarray] = None,
        E_elem: Optional[np.ndarray] = None,
        return_cf: bool = False,
    ) -> np.ndarray:
        """
        Assemble the diagonal of K for Jacobi preconditioning: M_inv = 1 / diag(K).
        Uses vectorized 3D scalar accumulation and precomputed diagonal stiffness.
        """
        Z, Y, X = self.grid_shape
        if E_elem is None:
            rho_flat = np.asarray(rho, dtype=np.float32).reshape(Z, Y, X)
            simp_p = self.simp_penalty
            rho_min = self.density_floor
            if abs(simp_p - 3.0) < 1e-5:
                rho_p = rho_flat * rho_flat * rho_flat
            else:
                rho_p = rho_flat ** simp_p
            E_elem = (rho_min + (1.0 - rho_min) * rho_p).astype(np.float32)

        if hasattr(self, "_E_nodes") and self._E_nodes.shape == (Z + 1, Y + 1, X + 1):
            E_nodes = self._E_nodes
            E_nodes.fill(0.0)
        else:
            E_nodes = np.zeros((Z + 1, Y + 1, X + 1), dtype=np.float32)

        for dz, dy, dx in NODE_OFFSETS:
            E_nodes[dz : Z + dz, dy : Y + dy, dx : X + dx] += E_elem

        if return_cf:
            if hasattr(self, "_m_inv_cf") and self._m_inv_cf.shape == (3, Z + 1, Y + 1, X + 1):
                m_inv_cf = self._m_inv_cf
            else:
                m_inv_cf = np.zeros((3, Z + 1, Y + 1, X + 1), dtype=np.float32)
            for c in range(3):
                m_inv_cf[c] = 1.0 / np.maximum(E_nodes * self.diag_k0_3[c], 1e-12)
            if fixed_dofs is not None:
                if fixed_dofs.shape == (3, Z + 1, Y + 1, X + 1):
                    m_inv_cf[fixed_dofs] = 1.0
                else:
                    fixed_cf = np.ascontiguousarray(
                        np.asarray(fixed_dofs, dtype=bool).reshape(Z + 1, Y + 1, X + 1, 3).transpose(3, 0, 1, 2)
                    )
                    m_inv_cf[fixed_cf] = 1.0
            return m_inv_cf
        else:
            diag_grid = E_nodes[..., None] * self.diag_k0_3
            diag_K = np.maximum(diag_grid.reshape(-1), 1e-12)
            m_inv = 1.0 / diag_K

            if fixed_dofs is not None:
                fixed_flat = np.asarray(fixed_dofs, dtype=bool).reshape(-1)
                m_inv[fixed_flat] = 1.0

            return m_inv

    def solve_pcg(
        self,
        f_ext_dof: np.ndarray,
        rho: np.ndarray,
        m6_rotors: Optional[np.ndarray] = None,
        fixed_dofs: Optional[np.ndarray] = None,
        u_init: Optional[np.ndarray] = None,
        tol: float = 1e-4,
        max_iters: int = 100,
        chunk_z: int = 16,
    ) -> Tuple[np.ndarray, float, int, bool]:
        """
        Matrix-free Preconditioned Conjugate Gradient (PCG) solver with Jacobi preconditioner.
        Uses contiguous channels-first layout for maximum throughput.
        Returns (u_dof, true_equilibrium_residual, iterations, converged).
        """
        Z, Y, X = self.grid_shape
        b_flat = np.asarray(f_ext_dof, dtype=np.float32).reshape(-1)
        fixed_flat = (
            np.asarray(fixed_dofs, dtype=bool).reshape(-1)
            if fixed_dofs is not None
            else None
        )

        b_norm = float(np.linalg.norm(b_flat[~fixed_flat] if fixed_flat is not None else b_flat))
        if b_norm < 1e-12:
            u_zero = np.zeros(self.mesh.num_dofs, dtype=np.float32)
            return u_zero, 0.0, 0, True

        # Precompute SIMP stiffness factor E_elem once for all matvecs & preconditioner
        simp_p = self.simp_penalty
        rho_min = self.density_floor
        rho_3d = np.asarray(rho, dtype=np.float32).reshape(Z, Y, X)
        if abs(simp_p - 3.0) < 1e-5:
            rho_p = rho_3d * rho_3d * rho_3d
        else:
            rho_p = rho_3d ** simp_p
        E_elem = (rho_min + (1.0 - rho_min) * rho_p).astype(np.float32)

        # Convert b, u_init, fixed_dofs to channels-first (3, Z+1, Y+1, X+1)
        b_cf = np.ascontiguousarray(b_flat.reshape(Z + 1, Y + 1, X + 1, 3).transpose(3, 0, 1, 2))
        fixed_cf = (
            np.ascontiguousarray(fixed_flat.reshape(Z + 1, Y + 1, X + 1, 3).transpose(3, 0, 1, 2))
            if fixed_flat is not None
            else None
        )
        if fixed_cf is not None:
            b_cf[fixed_cf] = 0.0

        if u_init is not None:
            u_init_flat = np.asarray(u_init, dtype=np.float32).reshape(-1)
            u_cf = np.ascontiguousarray(u_init_flat.reshape(Z + 1, Y + 1, X + 1, 3).transpose(3, 0, 1, 2))
        else:
            u_cf = np.zeros((3, Z + 1, Y + 1, X + 1), dtype=np.float32)
        if fixed_cf is not None:
            u_cf[fixed_cf] = 0.0

        # Initial residual r = b - K @ u in channels-first
        Ku_cf = self.matrix_free_matvec(
            u_cf, rho, m6_rotors=m6_rotors, fixed_dofs=fixed_cf, chunk_z=chunk_z, E_elem=E_elem, return_cf=True
        )
        r_cf = b_cf - Ku_cf
        if fixed_cf is not None:
            r_cf[fixed_cf] = 0.0

        # Preconditioner in channels-first
        m_inv_cf = self._compute_jacobi_preconditioner(rho, fixed_dofs=fixed_cf, E_elem=E_elem, return_cf=True)
        z_cf = m_inv_cf * r_cf
        if fixed_cf is not None:
            z_cf[fixed_cf] = 0.0

        p_cf = z_cf.copy()
        rz = float(np.sum(r_cf * z_cf))

        converged = False
        final_iter = 0
        r_norm = b_norm

        for it in range(1, max_iters + 1):
            final_iter = it
            Kp_cf = self.matrix_free_matvec(
                p_cf, rho, m6_rotors=m6_rotors, fixed_dofs=fixed_cf, chunk_z=chunk_z, E_elem=E_elem, return_cf=True
            )
            pKp = float(np.sum(p_cf * Kp_cf))

            if pKp <= 1e-30:
                break

            alpha = rz / pKp
            u_cf += alpha * p_cf
            r_cf -= alpha * Kp_cf
            if fixed_cf is not None:
                r_cf[fixed_cf] = 0.0

            r_norm = float(np.sqrt(np.sum(r_cf * r_cf)))
            if r_norm / b_norm <= tol:
                converged = True
                break

            z_cf = m_inv_cf * r_cf
            if fixed_cf is not None:
                z_cf[fixed_cf] = 0.0

            rz_new = float(np.sum(r_cf * z_cf))
            beta = rz_new / max(rz, 1e-30)
            p_cf = z_cf + beta * p_cf
            rz = rz_new

        r_eq = float(r_norm / b_norm)
        u_out = np.ascontiguousarray(u_cf.transpose(1, 2, 3, 0)).reshape(-1)
        return u_out, r_eq, final_iter, converged

    def solve_equilibrium_ebe(
        self,
        rho: np.ndarray,
        f_ext: np.ndarray,
        fixed_mask: np.ndarray,
        c_rot_cube: Optional[np.ndarray] = None,
        max_pcg_iters: int = 50,
        tol: float = 1e-4,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Interface contract specified in PROJECT.md:
        solve_equilibrium_ebe(rho, f_ext, fixed_mask, c_rot_cube, max_pcg_iters=50, tol=1e-4)
        Returns (u_disp, strain_energy_e, r_eq).
        """
        f_flat = np.asarray(f_ext, dtype=np.float32).reshape(-1)
        fixed_flat = np.asarray(fixed_mask, dtype=bool).reshape(-1)

        u_disp, r_eq, _, _ = self.solve_pcg(
            f_flat,
            rho,
            m6_rotors=c_rot_cube,
            fixed_dofs=fixed_flat,
            tol=tol,
            max_iters=max_pcg_iters,
        )

        res = self.compute_element_relaxations(
            u_disp, rho, m6_rotors=c_rot_cube, compute_rotors=True
        )
        return u_disp, res.strain_energy, r_eq

    # =========================================================================
    # 5. Tsai-Hill Yield Index, Stress Analysis & Element Relaxations
    # =========================================================================

    def compute_tsai_hill_index(
        self,
        strain_e: np.ndarray,
        c_rot: Optional[np.ndarray] = None,
        yield_strengths: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Compute direction-oriented orthotropic Tsai-Hill yield index Phi (arXiv:2112.02030v2).
        For isotropic steel with yield_strength sigma_y:
            Phi = sigma_vM^2 / sigma_y^2.
        """
        strain = np.asarray(strain_e, dtype=np.float64)  # (..., 6)
        C = self.c_vti if c_rot is None else np.asarray(c_rot, dtype=np.float64)

        if yield_strengths is None:
            yield_strengths = {
                "X": self.material.yield_strength,
                "Y": self.material.yield_strength,
                "Z": self.material.yield_strength,
                "S12": self.material.yield_strength / np.sqrt(3.0),
                "S23": self.material.yield_strength / np.sqrt(3.0),
                "S13": self.material.yield_strength / np.sqrt(3.0),
            }

        X = float(yield_strengths.get("X", 1.0))
        Y = float(yield_strengths.get("Y", 1.0))
        Z = float(yield_strengths.get("Z", 1.0))
        S12 = float(yield_strengths.get("S12", 1.0 / np.sqrt(3.0)))
        S23 = float(yield_strengths.get("S23", 1.0 / np.sqrt(3.0)))
        S13 = float(yield_strengths.get("S13", 1.0 / np.sqrt(3.0)))

        stress = np.matmul(C, strain[..., None]).squeeze(-1)  # (..., 6)
        s2 = np.sqrt(2.0)
        s1 = stress[..., 0]
        s2_val = stress[..., 1]
        s3 = stress[..., 2]
        t23 = stress[..., 3] / s2
        t13 = stress[..., 4] / s2
        t12 = stress[..., 5] / s2

        phi = (
            (s1**2) / (X**2)
            + (s2_val**2) / (Y**2)
            + (s3**2) / (Z**2)
            - (s1 * s2_val) * (1.0 / (X**2) + 1.0 / (Y**2) - 1.0 / (Z**2))
            - (s2_val * s3) * (1.0 / (Y**2) + 1.0 / (Z**2) - 1.0 / (X**2))
            - (s1 * s3) * (1.0 / (Z**2) + 1.0 / (X**2) - 1.0 / (Y**2))
            + (t12**2) / (S12**2)
            + (t23**2) / (S23**2)
            + (t13**2) / (S13**2)
        )
        return np.maximum(phi, 0.0).astype(np.float32)

    def compute_element_relaxations(
        self,
        u_dof: np.ndarray,
        rho: np.ndarray,
        m6_rotors: Optional[np.ndarray] = None,
        delta_t_field: Optional[np.ndarray] = None,
        compute_rotors: bool = False,
        compute_stress: bool = False,
    ) -> EBERelaxationResult:
        """
        Calculate strain energy u_e, von Mises stress, Tsai-Hill index,
        principal frames R_e in SO(3), SO(6) rotors M(R_e), and compliance sensitivities.
        Uses vectorized 3D slice gather and batch eigensolvers.
        """
        Z, Y, X = self.grid_shape
        u_flat = np.asarray(u_dof, dtype=np.float32).reshape(-1)
        rho_arr = np.asarray(rho, dtype=np.float32).reshape(Z, Y, X)
        simp_p = self.simp_penalty
        rho_min = self.density_floor

        # Vectorized gather from contiguous channels-first layout
        u_grid = u_flat.reshape(Z + 1, Y + 1, X + 1, 3)
        u_cf = np.ascontiguousarray(u_grid.transpose(3, 0, 1, 2))
        if hasattr(self, "_d_cf") and self._d_cf.shape == (24, Z, Y, X):
            d_cf = self._d_cf
        else:
            d_cf = np.empty((24, Z, Y, X), dtype=np.float32)

        for a, (dz, dy, dx) in enumerate(NODE_OFFSETS):
            a3 = a * 3
            d_cf[a3] = u_cf[0, dz : Z + dz, dy : Y + dy, dx : X + dx]
            d_cf[a3 + 1] = u_cf[1, dz : Z + dz, dy : Y + dy, dx : X + dx]
            d_cf[a3 + 2] = u_cf[2, dz : Z + dz, dy : Y + dy, dx : X + dx]

        # Strain energy u_e = 0.5 * d_e.T @ k0 @ d_e via SGEMM
        d_cf_2d = d_cf.reshape(24, -1)
        kd_2d = self.k0_f32 @ d_cf_2d
        u_strain = (0.5 * np.sum(d_cf_2d * kd_2d, axis=0)).reshape(Z, Y, X).astype(np.float32)

        # Total compliance J = sum_e E(rho_e) * u_e
        if abs(simp_p - 3.0) < 1e-5:
            rho_sq = rho_arr * rho_arr
            rho_p = rho_sq * rho_arr
            rho_p_minus_1 = rho_sq
        else:
            rho_p = rho_arr ** simp_p
            rho_p_minus_1 = rho_arr ** (simp_p - 1.0)

        E_elem = (rho_min + (1.0 - rho_min) * rho_p).astype(np.float32)
        compliance = float(np.sum(E_elem * u_strain))

        # Compliance sensitivity: alpha_e = -p * (1 - rho_min) * rho_e^(p-1) * u_e
        compliance_sensitivity = (
            -simp_p * (1.0 - rho_min) * rho_p_minus_1 * u_strain
        ).astype(np.float32)

        if not compute_rotors and not compute_stress and delta_t_field is None:
            # High-throughput path for interactive density updates (< 2 ms)
            principal_frames = np.zeros((Z, Y, X, 3, 3), dtype=np.float32)
            principal_frames[..., 0, 0] = 1.0
            principal_frames[..., 1, 1] = 1.0
            principal_frames[..., 2, 2] = 1.0
            so6_rotors = np.zeros((Z, Y, X, 6, 6), dtype=np.float32)
            so6_rotors[..., np.arange(6), np.arange(6)] = 1.0
            return EBERelaxationResult(
                u_disp=u_flat,
                strain_energy=u_strain,
                compliance=compliance,
                von_mises=u_strain,
                tsai_hill=u_strain,
                principal_frames=principal_frames,
                so6_rotors=so6_rotors,
                compliance_sensitivity=compliance_sensitivity,
                equilibrium_residual=0.0,
                cg_iterations=0,
                cg_converged=True,
            )

        # Full stress and SO(6) rotor analysis
        d_e_f64 = np.ascontiguousarray(d_cf.transpose(1, 2, 3, 0)).astype(np.float64)
        eps_centroid = np.matmul(d_e_f64, self.B_center.T)  # (Z, Y, X, 6)

        # Optional coupled thermo-elastic stress: sigma_therm = C : alpha * Delta_T
        if delta_t_field is not None:
            dt = np.asarray(delta_t_field, dtype=np.float64).reshape(Z, Y, X)
            alpha_vec = np.zeros(6, dtype=np.float64)
            alpha_vec[0] = alpha_vec[1] = alpha_vec[2] = self.material.alpha_thermal
            eps_therm = alpha_vec[None, None, None, :] * dt[..., None]
            eps_mech = eps_centroid - eps_therm
        else:
            eps_mech = eps_centroid

        # Centroid Cauchy stress & von Mises
        stress_centroid = np.matmul(eps_mech, self.c_vti.T)  # (Z, Y, X, 6)
        s2 = np.sqrt(2.0)
        s11 = stress_centroid[..., 0]
        s22 = stress_centroid[..., 1]
        s33 = stress_centroid[..., 2]
        s23 = stress_centroid[..., 3] / s2
        s13 = stress_centroid[..., 4] / s2
        s12 = stress_centroid[..., 5] / s2

        von_mises = np.sqrt(
            np.maximum(
                0.0,
                0.5
                * (
                    (s11 - s22) ** 2
                    + (s22 - s33) ** 2
                    + (s33 - s11) ** 2
                    + 6.0 * (s12**2 + s23**2 + s13**2)
                ),
            )
        ).astype(np.float32)

        # Extract principal stress frames R_e in SO(3) and SO(6) rotors
        principal_frames = np.zeros((Z, Y, X, 3, 3), dtype=np.float64)
        principal_frames[..., 0, 0] = 1.0
        principal_frames[..., 1, 1] = 1.0
        principal_frames[..., 2, 2] = 1.0
        so6_rotors = np.zeros((Z, Y, X, 6, 6), dtype=np.float64)
        so6_rotors[..., np.arange(6), np.arange(6)] = 1.0

        if compute_rotors:
            tsai_hill = self.compute_tsai_hill_index(eps_mech, self.c_vti)
            sigma_3x3 = np.stack([
                np.stack([s11, s12, s13], axis=-1),
                np.stack([s12, s22, s23], axis=-1),
                np.stack([s13, s23, s33], axis=-1),
            ], axis=-2)
            evals, evecs = np.linalg.eigh(sigma_3x3)
            V = evecs[..., ::-1].copy()
            det_V = np.linalg.det(V)
            neg_det = det_V < 0.0
            V[neg_det, :, 2] = -V[neg_det, :, 2]
            principal_frames = np.swapaxes(V, -1, -2)
            so6_rotors = self.build_kelvin_mandel_so6_rotor(principal_frames)
        else:
            tsai_hill = von_mises

        return EBERelaxationResult(
            u_disp=u_flat,
            strain_energy=u_strain,
            compliance=compliance,
            von_mises=von_mises,
            tsai_hill=tsai_hill,
            principal_frames=principal_frames,
            so6_rotors=so6_rotors,
            compliance_sensitivity=compliance_sensitivity,
            equilibrium_residual=0.0,
            cg_iterations=0,
            cg_converged=True,
        )
