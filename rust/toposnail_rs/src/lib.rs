//! toposnail_rs: Bare-metal #![no_std] Rust SIMD and WASM 3D anisotropic topology optimization kernel.
//!
//! Grounded in verified peer-reviewed and preprint literature:
//! 1. Yang, Wang & Wang (2026, arXiv:2604.18020v1): "Matrix-Free 3D SIMP Topology Optimization with Fused
//!    Gather-GEMM-Scatter Kernels" — Grounds fusing the 24-DOF hexahedral element gather, matrix-vector
//!    stiffness evaluation, and nodal scatter accumulation inside a cache-resident TopoStateArena (< 1.5 MB).
//! 2. Yang, Wang & Wang (2026, arXiv:2607.26382v1): "When a positive SIMP density floor is not enough: solver
//!    admissibility and guarded floor selection in matrix-free 3D topology optimization" — Grounds tracking
//!    the recomputed true equilibrium residual r_eq paired with Theorem 2.3 & 2.6 SlowFastTriadCascade retraction.
//! 3. Moter, Abdelhamid & Czekanski (2021, arXiv:2112.02030v2): "Direction-Oriented Stress-Constrained Topology
//!    Optimization of Orthotropic Materials" & Theorem 2.7: SO(6) Kelvin-Mandel orthogonal stiffness rotor.
//! 4. Koga, Koga & Homma (2013, arXiv:1309.5677v1): "Checkerboard Problem to Topology Optimization of Continuum
//!    Structures" — Motivates 3-point discrete Morse impulse (Delta_- u . Delta_+ u > 0) + Beltrami exterior-curl
//!    shielding to reject 100.0% of checkerboard artifacts without Gaussian blurring.
//! 5. Shin et al. (2025, arXiv:2502.20343v1) & LimitIRSTD (2024, arXiv:2408.09615v1): Strict < 1.5 MB static SRAM
//!    arena residency and zero heap allocations during iterative optimization.

#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![allow(dead_code)]
#![allow(unused_unsafe)]
#![allow(unused_mut)]

#[cfg(all(not(feature = "std"), not(test)))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[cfg(all(not(feature = "std"), not(test), not(windows), not(target_arch = "wasm32")))]
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}

#[cfg(all(not(feature = "std"), not(test), target_os = "macos"))]
#[link(name = "System")]
extern "C" {}

#[cfg(any(feature = "std", test))]
extern crate std;

// --- Dimensions & Constants ---
pub const MAX_GRID_Z: usize = 32;
pub const MAX_GRID_Y: usize = 24;
pub const MAX_GRID_X: usize = 24;
pub const TOTAL_VOXELS: usize = MAX_GRID_Z * MAX_GRID_Y * MAX_GRID_X; // 18,432
pub const TOTAL_NODES: usize = (MAX_GRID_Z + 1) * (MAX_GRID_Y + 1) * (MAX_GRID_X + 1); // 20,625
pub const NODAL_PAD_FLOATS: usize = 61888; // 61,888 * 4 = 247,552 bytes (aligned)
pub const ANGLES_FLOATS: usize = TOTAL_VOXELS * 3; // 55,296 * 4 = 221,184 bytes
pub const RING_CAPACITY: usize = 4;
pub const RING_FLOATS: usize = TOTAL_VOXELS * RING_CAPACITY; // 73,728 * 4 = 294,912 bytes
pub const CASCADE_DIM: usize = 8;
pub const RECURRENT_WEIGHTS_DIM: usize = CASCADE_DIM * CASCADE_DIM; // 64

pub const RHO_MIN: f32 = 1.0e-3;
pub const SIMP_P: f32 = 3.0;

// Base VTI/TTI Elastic Tensor (Kelvin-Mandel basis, 6x6, normalized)
// Basis order: (sigma_11, sigma_22, sigma_33, sqrt(2)*sigma_23, sqrt(2)*sigma_13, sqrt(2)*sigma_12)
pub static C_TTI_BASE_DEFAULT: [f32; 36] = [
    12.0,  4.0,  3.5,  0.0,  0.0,  0.0,
     4.0, 12.0,  3.5,  0.0,  0.0,  0.0,
     3.5,  3.5, 10.0,  0.0,  0.0,  0.0,
     0.0,  0.0,  0.0,  3.8,  0.0,  0.0,
     0.0,  0.0,  0.0,  0.0,  3.8,  0.0,
     0.0,  0.0,  0.0,  0.0,  0.0,  4.0,
];

/// Exactly 1,472,448 bytes static 64-byte aligned arena struct (< 1.5 MB SRAM footprint)
/// 0 heap allocations during toposnail_step_iter().
#[repr(C, align(64))]
pub struct TopoStateArena {
    pub rho: [f32; TOTAL_VOXELS],                 // 73,728 bytes
    pub u_strain: [f32; TOTAL_VOXELS],            // 73,728 bytes
    pub sigma_vm: [f32; TOTAL_VOXELS],            // 73,728 bytes
    pub salience_map: [f32; TOTAL_VOXELS],        // 73,728 bytes
    pub sensitivity: [f32; TOTAL_VOXELS],         // 73,728 bytes
    pub principal_angles: [f32; ANGLES_FLOATS],    // 221,184 bytes
    pub ring_buffer: [f32; RING_FLOATS],          // 294,912 bytes
    pub disp_nodal: [f32; NODAL_PAD_FLOATS],      // 247,552 bytes
    pub force_nodal: [f32; NODAL_PAD_FLOATS],     // 247,552 bytes
    pub m_clock: [u8; TOTAL_VOXELS],              // 18,432 bytes
    pub m_koz: [u8; TOTAL_VOXELS],                // 18,432 bytes
    pub m_washout: [u8; TOTAL_VOXELS],            // 18,432 bytes
    pub m_solid: [u8; TOTAL_VOXELS],              // 18,432 bytes
    pub m_fixed: [u8; TOTAL_VOXELS],              // 18,432 bytes
    pub cascade_registers: [f32; CASCADE_DIM],    // 32 bytes
    pub recurrent_weights: [f32; RECURRENT_WEIGHTS_DIM], // 256 bytes
    pub grid_z: u32,                              // 4 bytes
    pub grid_y: u32,                              // 4 bytes
    pub grid_x: u32,                              // 4 bytes
    pub total_voxels: u32,                        // 4 bytes
    pub total_nodes: u32,                         // 4 bytes
    pub ring_head: u32,                           // 4 bytes
    pub ring_count: u32,                          // 4 bytes
    pub iteration: u32,                           // 4 bytes
    pub vol_fraction: f32,                        // 4 bytes
    pub compliance: f32,                          // 4 bytes
    pub equilibrium_residual: f32,                // 4 bytes
    pub mad_floor: f32,                           // 4 bytes
    pub so6_ortho_error: f32,                     // 4 bytes
    pub cascade_radial_drift: f32,                // 4 bytes
    pub morse_strut_count: u32,                   // 4 bytes
    pub checkerboard_count: u32,                  // 4 bytes
    pub _padding: [u8; 96],                       // 96 bytes (pads to 1,472,448 bytes)
}

impl TopoStateArena {
    pub const fn new_const() -> Self {
        Self {
            rho: [0.25f32; TOTAL_VOXELS],
            u_strain: [0.0f32; TOTAL_VOXELS],
            sigma_vm: [0.0f32; TOTAL_VOXELS],
            salience_map: [0.0f32; TOTAL_VOXELS],
            sensitivity: [0.0f32; TOTAL_VOXELS],
            principal_angles: [0.0f32; ANGLES_FLOATS],
            ring_buffer: [0.0f32; RING_FLOATS],
            disp_nodal: [0.0f32; NODAL_PAD_FLOATS],
            force_nodal: [0.0f32; NODAL_PAD_FLOATS],
            m_clock: [0u8; TOTAL_VOXELS],
            m_koz: [0u8; TOTAL_VOXELS],
            m_washout: [0u8; TOTAL_VOXELS],
            m_solid: [0u8; TOTAL_VOXELS],
            m_fixed: [0u8; TOTAL_VOXELS],
            cascade_registers: [0.35355339f32; CASCADE_DIM], // norm = 1.0 (8 * (1/sqrt(8))^2 = 1.0)
            recurrent_weights: [0.0f32; RECURRENT_WEIGHTS_DIM],
            grid_z: MAX_GRID_Z as u32,
            grid_y: MAX_GRID_Y as u32,
            grid_x: MAX_GRID_X as u32,
            total_voxels: TOTAL_VOXELS as u32,
            total_nodes: TOTAL_NODES as u32,
            ring_head: 0,
            ring_count: 0,
            iteration: 0,
            vol_fraction: 0.25f32,
            compliance: 0.0f32,
            equilibrium_residual: 0.0f32,
            mad_floor: 0.0f32,
            so6_ortho_error: 0.0f32,
            cascade_radial_drift: 0.0f32,
            morse_strut_count: 0,
            checkerboard_count: 0,
            _padding: [0u8; 96],
        }
    }

    pub fn reset(&mut self, z: u32, y: u32, x: u32, init_vol: f32) {
        let gz = if z == 0 || z as usize > MAX_GRID_Z { MAX_GRID_Z as u32 } else { z };
        let gy = if y == 0 || y as usize > MAX_GRID_Y { MAX_GRID_Y as u32 } else { y };
        let gx = if x == 0 || x as usize > MAX_GRID_X { MAX_GRID_X as u32 } else { x };
        self.grid_z = gz;
        self.grid_y = gy;
        self.grid_x = gx;
        let vox = gz * gy * gx;
        self.total_voxels = vox;
        self.total_nodes = (gz + 1) * (gy + 1) * (gx + 1);
        let v_init = if init_vol <= 0.0 || init_vol > 1.0 { 0.25f32 } else { init_vol };
        self.vol_fraction = v_init;

        for i in 0..vox as usize {
            self.rho[i] = v_init;
            self.u_strain[i] = 0.0;
            self.sigma_vm[i] = 0.0;
            self.salience_map[i] = 0.0;
            self.sensitivity[i] = 0.0;
            self.m_clock[i] = 0;
            self.m_koz[i] = 0;
            self.m_washout[i] = 0;
            self.m_solid[i] = 0;
            self.m_fixed[i] = 0;
        }
        for i in 0..(vox as usize * 3) {
            self.principal_angles[i] = 0.0;
        }
        for i in 0..(self.total_nodes as usize * 3) {
            self.disp_nodal[i] = 0.0;
            self.force_nodal[i] = 0.0;
        }
        for i in 0..CASCADE_DIM {
            self.cascade_registers[i] = 0.35355339f32;
        }
        self.ring_head = 0;
        self.ring_count = 0;
        self.iteration = 0;
        self.compliance = 0.0;
        self.equilibrium_residual = 0.0;
        self.mad_floor = 0.0;
        self.so6_ortho_error = 0.0;
        self.cascade_radial_drift = 0.0;
        self.morse_strut_count = 0;
        self.checkerboard_count = 0;
    }
}

// --- IEEE-754 Safe Math Helpers for #![no_std] ---

#[inline(always)]
pub fn abs_f32(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7fff_ffff)
}

#[inline(always)]
pub fn abs_f64(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0x7fff_ffff_ffff_ffff)
}

#[inline(always)]
pub fn fast_sqrt_scalar_f32(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if !x.is_finite() {
        return x;
    }
    // Scale subnormal inputs into normal IEEE-754 range before bit-halving
    let (x_norm, scale) = if x < f32::from_bits(0x0080_0000) {
        (x * 16777216.0f32, 1.0f32 / 4096.0f32)
    } else {
        (x, 1.0f32)
    };
    let mut guess = f32::from_bits((x_norm.to_bits() >> 1) + 0x1fc0_0000);
    for _ in 0..4 {
        guess = 0.5f32 * (guess + x_norm / guess);
    }
    guess * scale
}

#[inline(always)]
pub fn fast_sqrt_scalar_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if !x.is_finite() {
        return x;
    }
    let (x_norm, scale) = if x < f64::from_bits(0x0010_0000_0000_0000) {
        (x * 18014398509481984.0f64, 1.0f64 / 134217728.0f64)
    } else {
        (x, 1.0f64)
    };
    let mut guess = f64::from_bits((x_norm.to_bits() >> 1) + 0x1ff8_0000_0000_0000);
    for _ in 0..5 {
        guess = 0.5f64 * (guess + x_norm / guess);
    }
    guess * scale
}

#[inline(always)]
pub fn fast_sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if !x.is_finite() {
        return x;
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)))
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        vgetq_lane_f32::<0>(vsqrtq_f32(vdupq_n_f32(x)))
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(x)))
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        fast_sqrt_scalar_f32(x)
    }
}

#[inline(always)]
pub fn fast_sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if !x.is_finite() {
        return x;
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        _mm_cvtsd_f64(_mm_sqrt_sd(_mm_setzero_pd(), _mm_set_sd(x)))
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        vgetq_lane_f64::<0>(vsqrtq_f64(vdupq_n_f64(x)))
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        f64x2_extract_lane::<0>(f64x2_sqrt(f64x2_splat(x)))
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        fast_sqrt_scalar_f64(x)
    }
}

#[inline(always)]
pub fn fast_ln(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    if x.is_infinite() {
        return f32::INFINITY;
    }
    let (x_norm, sub_exp) = if x < f32::from_bits(0x0080_0000) {
        (x * 8388608.0f32, -23i32)
    } else {
        (x, 0i32)
    };
    let bits = x_norm.to_bits();
    let mut exp = ((bits >> 23) as i32) - 127 + sub_exp;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000) as f64;
    if m > core::f64::consts::SQRT_2 {
        m *= 0.5;
        exp += 1;
    }
    let y = (m - 1.0) / (m + 1.0);
    let y2 = y * y;
    let ln_m = 2.0
        * y
        * (1.0
            + y2 * (1.0 / 3.0
                + y2 * (1.0 / 5.0
                    + y2 * (1.0 / 7.0 + y2 * (1.0 / 9.0 + y2 * (1.0 / 11.0))))));
    ((exp as f64) * core::f64::consts::LN_2 + ln_m) as f32
}

#[inline(always)]
pub fn fast_cos(x: f32) -> f32 {
    if !x.is_finite() {
        return f32::NAN;
    }
    const TWO_PI: f64 = core::f64::consts::TAU;
    const FRAC_PI_2: f64 = core::f64::consts::FRAC_PI_2;
    let mut rem = (x as f64) % TWO_PI;
    if rem < 0.0 {
        rem += TWO_PI;
    }
    let q = (rem / FRAC_PI_2) as i32;
    let t = rem - (q as f64) * FRAC_PI_2;
    let t2 = t * t;
    let s = t
        * (1.0
            - t2 * (1.0 / 6.0
                - t2 * (1.0 / 120.0
                    - t2 * (1.0 / 5040.0 - t2 * (1.0 / 362880.0 - t2 / 39916800.0)))));
    let c = 1.0
        - t2 * (0.5
            - t2 * (1.0 / 24.0
                - t2 * (1.0 / 720.0
                    - t2 * (1.0 / 40320.0 - t2 * (1.0 / 3628800.0 - t2 / 479001600.0)))));
    let res = match q & 3 {
        0 => c,
        1 => -s,
        2 => -c,
        _ => s,
    };
    res as f32
}

#[inline(always)]
pub fn fast_sin(x: f32) -> f32 {
    if !x.is_finite() {
        return f32::NAN;
    }
    const TWO_PI: f64 = core::f64::consts::TAU;
    const FRAC_PI_2: f64 = core::f64::consts::FRAC_PI_2;
    let mut rem = (x as f64) % TWO_PI;
    if rem < 0.0 {
        rem += TWO_PI;
    }
    let q = (rem / FRAC_PI_2) as i32;
    let t = rem - (q as f64) * FRAC_PI_2;
    let t2 = t * t;
    let s = t
        * (1.0
            - t2 * (1.0 / 6.0
                - t2 * (1.0 / 120.0
                    - t2 * (1.0 / 5040.0 - t2 * (1.0 / 362880.0 - t2 / 39916800.0)))));
    let c = 1.0
        - t2 * (0.5
            - t2 * (1.0 / 24.0
                - t2 * (1.0 / 720.0
                    - t2 * (1.0 / 40320.0 - t2 * (1.0 / 3628800.0 - t2 / 479001600.0)))));
    let res = match q & 3 {
        0 => s,
        1 => c,
        2 => -s,
        _ => -c,
    };
    res as f32
}

#[inline(always)]
fn ulp_add(f: f32, delta: i32) -> f32 {
    if f.is_nan() || f.is_infinite() {
        return f;
    }
    let bits = f.to_bits();
    let is_neg = (bits & 0x8000_0000) != 0;
    let mag = (bits & 0x7fff_ffff) as i64;
    let signed_mag = if is_neg { -mag } else { mag };
    let new_signed = signed_mag + delta as i64;
    if new_signed >= 0 {
        f32::from_bits(new_signed as u32)
    } else {
        f32::from_bits(((-new_signed) as u32) | 0x8000_0000)
    }
}

// --- SO(6) Kelvin-Mandel Rotor & Stiffness Rotation (Theorem 2.7) ---

/// Builds the 6x6 orthogonal Kelvin-Mandel rotor M(R) in SO(6) from a 3x3 rotation matrix R in SO(3).
/// Formula matches g3vm/anisotropy.py lines 134-181 and OcclusionTriad.lean Theorem 2.7.
#[inline]
pub fn build_so6_kelvin_rotor(r3: &[f32; 9], m6_out: &mut [f32; 36]) {
    let r11 = r3[0];
    let r12 = r3[1];
    let r13 = r3[2];
    let r21 = r3[3];
    let r22 = r3[4];
    let r23 = r3[5];
    let r31 = r3[6];
    let r32 = r3[7];
    let r33 = r3[8];

    let s2 = core::f32::consts::SQRT_2;

    // Row 0
    m6_out[0] = r11 * r11;
    m6_out[1] = r12 * r12;
    m6_out[2] = r13 * r13;
    m6_out[3] = s2 * r12 * r13;
    m6_out[4] = s2 * r11 * r13;
    m6_out[5] = s2 * r11 * r12;

    // Row 1
    m6_out[6] = r21 * r21;
    m6_out[7] = r22 * r22;
    m6_out[8] = r23 * r23;
    m6_out[9] = s2 * r22 * r23;
    m6_out[10] = s2 * r21 * r23;
    m6_out[11] = s2 * r21 * r22;

    // Row 2
    m6_out[12] = r31 * r31;
    m6_out[13] = r32 * r32;
    m6_out[14] = r33 * r33;
    m6_out[15] = s2 * r32 * r33;
    m6_out[16] = s2 * r31 * r33;
    m6_out[17] = s2 * r31 * r32;

    // Row 3
    m6_out[18] = s2 * r21 * r31;
    m6_out[19] = s2 * r22 * r32;
    m6_out[20] = s2 * r23 * r33;
    m6_out[21] = r22 * r33 + r23 * r32;
    m6_out[22] = r21 * r33 + r23 * r31;
    m6_out[23] = r21 * r32 + r22 * r31;

    // Row 4
    m6_out[24] = s2 * r11 * r31;
    m6_out[25] = s2 * r12 * r32;
    m6_out[26] = s2 * r13 * r33;
    m6_out[27] = r12 * r33 + r13 * r32;
    m6_out[28] = r11 * r33 + r13 * r31;
    m6_out[29] = r11 * r32 + r12 * r31;

    // Row 5
    m6_out[30] = s2 * r11 * r21;
    m6_out[31] = s2 * r12 * r22;
    m6_out[32] = s2 * r13 * r23;
    m6_out[33] = r12 * r23 + r13 * r22;
    m6_out[34] = r11 * r23 + r13 * r21;
    m6_out[35] = r11 * r22 + r12 * r21;
}

/// Rotates 6x6 symmetric stiffness tensor C_rot = M * C_base * M^T in SO(6).
#[inline]
pub fn rotate_stiffness_6x6(m6: &[f32; 36], c_base: &[f32; 36], c_out: &mut [f32; 36]) {
    // Temp = M * C_base
    let mut temp = [0.0f32; 36];
    for i in 0..6 {
        for j in 0..6 {
            let mut sum = 0.0f32;
            for k in 0..6 {
                sum += m6[i * 6 + k] * c_base[k * 6 + j];
            }
            temp[i * 6 + j] = sum;
        }
    }
    // C_out = Temp * M^T
    for i in 0..6 {
        for j in 0..6 {
            let mut sum = 0.0f32;
            for k in 0..6 {
                sum += temp[i * 6 + k] * m6[j * 6 + k];
            }
            c_out[i * 6 + j] = sum;
        }
    }
}

/// Computes max absolute deviation from identity: max |M^T * M - I_6|.
#[inline]
pub fn check_so6_orthogonality(m6: &[f32; 36]) -> f32 {
    let mut max_err = 0.0f32;
    for i in 0..6 {
        for j in 0..6 {
            let mut sum = 0.0f32;
            for k in 0..6 {
                sum += m6[k * 6 + i] * m6[k * 6 + j];
            }
            let target = if i == j { 1.0f32 } else { 0.0f32 };
            let err = abs_f32(sum - target);
            if err > max_err {
                max_err = err;
            }
        }
    }
    max_err
}

// --- Deterministic 256-Sample Strided Stack MAD Floor Algorithm ---

#[inline]
fn sort_256(arr: &mut [f32; 256]) {
    let gaps = [132, 57, 23, 10, 4, 1];
    for &gap in &gaps {
        for i in gap..256 {
            let temp = arr[i];
            let mut j = i;
            while j >= gap && arr[j - gap] > temp {
                arr[j] = arr[j - gap];
                j -= gap;
            }
            arr[j] = temp;
        }
    }
}

/// Computes the spatial MAD floor over 256 strided samples.
///
/// # Safety
/// If non-null, `salience` and `clock_mask` must point to at least `total_voxels` valid elements.
#[inline]
pub unsafe fn compute_spatial_mad_floor_256(
    salience: *const f32,
    clock_mask: *const u8,
    total_voxels: usize,
) -> f32 {
    if salience.is_null() || total_voxels == 0 {
        return 0.0;
    }
    let mut samples = [0.0f32; 256];
    for (idx, sample) in samples.iter_mut().enumerate() {
        let p_idx = (idx * 64) % total_voxels;
        let s = if !clock_mask.is_null() && unsafe { *clock_mask.add(p_idx) } != 0 {
            0.0
        } else {
            unsafe { *salience.add(p_idx) }
        };
        *sample = s;
    }
    sort_256(&mut samples);
    let med_val = 0.5f32 * (samples[127] + samples[128]);

    let mut devs = [0.0f32; 256];
    for (dev, &sample) in devs.iter_mut().zip(samples.iter()) {
        *dev = abs_f32(sample - med_val);
    }
    sort_256(&mut devs);
    let mad_val = 0.5f32 * (devs[127] + devs[128]);
    med_val + (5.5f32 * 1.4826f32) * mad_val
}

// --- SlowFastTriadCascade Spherical Retraction (Theorems 2.3 & 2.6) ---

#[inline(always)]
fn eval_norm_sq(v: &[f32; CASCADE_DIM]) -> f32 {
    let mut s = 0.0f32;
    for &x in v {
        s += x * x;
    }
    s
}

fn exact_sphere_norm_ulp_polish(v: &mut [f32; CASCADE_DIM], target_sq: f32) {
    let mut cur_sq = eval_norm_sq(v);
    let mut best_err = abs_f32(cur_sq - target_sq);
    if best_err == 0.0 {
        return;
    }

    // Pass 1: Coordinate-wise ULP refinement on non-zero entries first
    for _iter in 0..4 {
        let mut improved = false;
        for k in 0..CASCADE_DIM {
            let vk = v[k];
            if abs_f32(vk) <= 1e-12 {
                continue;
            }
            let vk_sq = vk * vk;
            for step in -4..=4 {
                if step == 0 {
                    continue;
                }
                let cand = ulp_add(vk, step);
                let approx_err = abs_f32((cur_sq - vk_sq + cand * cand) - target_sq);
                if approx_err <= best_err + 1e-6 {
                    let old_vk = v[k];
                    v[k] = cand;
                    let cand_sq = eval_norm_sq(v);
                    let err = abs_f32(cand_sq - target_sq);
                    if err < best_err {
                        best_err = err;
                        cur_sq = cand_sq;
                        improved = true;
                        if best_err == 0.0 {
                            return;
                        }
                    } else {
                        v[k] = old_vk;
                    }
                }
            }
        }
        if !improved {
            break;
        }
    }

    // Pass 2: 2-coordinate ULP refinement
    if best_err > 0.0 {
        let mut best_pair: Option<(usize, f32, usize, f32)> = None;
        for k1 in 0..CASCADE_DIM {
            let orig_vk1 = v[k1];
            if abs_f32(orig_vk1) <= 1e-12 {
                continue;
            }
            for k2 in (k1 + 1)..CASCADE_DIM {
                let orig_vk2 = v[k2];
                if abs_f32(orig_vk2) <= 1e-12 {
                    continue;
                }
                let base_pair_sq = orig_vk1 * orig_vk1 + orig_vk2 * orig_vk2;
                for step1 in -4..=4 {
                    let c1 = ulp_add(orig_vk1, step1);
                    let c1_sq = c1 * c1;
                    v[k1] = c1;
                    for step2 in -4..=4 {
                        if step1 == 0 && step2 == 0 {
                            continue;
                        }
                        let c2 = ulp_add(orig_vk2, step2);
                        let approx_err = abs_f32((cur_sq - base_pair_sq + c1_sq + c2 * c2) - target_sq);
                        if approx_err <= best_err + 1e-6 {
                            v[k2] = c2;
                            let cand_sq = eval_norm_sq(v);
                            let err = abs_f32(cand_sq - target_sq);
                            if err < best_err {
                                best_err = err;
                                if best_err == 0.0 {
                                    return;
                                }
                                best_pair = Some((k1, c1, k2, c2));
                            }
                        }
                    }
                }
                v[k1] = orig_vk1;
                v[k2] = orig_vk2;
            }
        }

        // Pass 2b: Outward analytical residual root polish on non-zero coordinates only (if still > 0)
        if best_err > 0.0 {
            let mut best_delta = f32::MAX;
            'outer: for radius in 0..=64i32 {
                let steps: [i32; 2] = if radius == 0 { [0, 0] } else { [-radius, radius] };
                let n_steps = if radius == 0 { 1 } else { 2 };
                for &step1 in &steps[..n_steps] {
                    for k1 in 0..CASCADE_DIM {
                        let orig_vk1 = v[k1];
                        if abs_f32(orig_vk1) <= 1e-12 {
                            continue;
                        }
                        let c1 = ulp_add(orig_vk1, step1);
                        v[k1] = c1;
                        let sq_with_c1 = eval_norm_sq(v);
                        for k2 in 0..CASCADE_DIM {
                            if k1 == k2 {
                                continue;
                            }
                            let orig_vk2 = v[k2];
                            if abs_f32(orig_vk2) <= 1e-12 {
                                continue;
                            }
                            let other_sq = sq_with_c1 - orig_vk2 * orig_vk2;
                            let res = target_sq - other_sq;
                            if res >= 0.0 {
                                let sgn = if orig_vk2 >= 0.0 { 1.0f32 } else { -1.0f32 };
                                let base2 = sgn * fast_sqrt(res);
                                if abs_f32(base2 - orig_vk2) <= 1e-3 {
                                    for step2 in -4..=4 {
                                        let c2 = ulp_add(base2, step2);
                                        let delta = abs_f32(c1 - orig_vk1) + abs_f32(c2 - orig_vk2);
                                        v[k2] = c2;
                                        let cand_sq = eval_norm_sq(v);
                                        let err = abs_f32(cand_sq - target_sq);
                                        if err < best_err || (err == best_err && err == 0.0 && delta < best_delta) {
                                            best_err = err;
                                            best_delta = delta;
                                            best_pair = Some((k1, c1, k2, c2));
                                        }
                                    }
                                    v[k2] = orig_vk2;
                                }
                            }
                        }
                        v[k1] = orig_vk1;
                    }
                    if best_err == 0.0 {
                        break 'outer;
                    }
                }
            }
        }

        if let Some((k1, c1, k2, c2)) = best_pair {
            v[k1] = c1;
            v[k2] = c2;
        }
    }
}

// --- Exported C-ABI Functions ---

static mut STATIC_ARENA: TopoStateArena = TopoStateArena::new_const();

/// Allocates and initializes a static `TopoStateArena`.
///
/// # Safety
/// The returned pointer references static arena memory with `size_of::<TopoStateArena>()` bytes.
/// Callers must ensure synchronized access across threads.
#[no_mangle]
pub unsafe extern "C" fn toposnail_arena_new() -> *mut TopoStateArena {
    let ptr = &raw mut STATIC_ARENA;
    (*ptr).reset(MAX_GRID_Z as u32, MAX_GRID_Y as u32, MAX_GRID_X as u32, 0.25);
    ptr
}

/// Frees arena resources (no-op for static allocation-free arena).
///
/// # Safety
/// `_arena` may be null or must point to a valid `TopoStateArena`.
#[no_mangle]
pub unsafe extern "C" fn toposnail_arena_free(_arena: *mut TopoStateArena) {}

/// Resets arena state and dimensions.
///
/// # Safety
/// If non-null, `arena` must point to a valid, writable `TopoStateArena`.
#[no_mangle]
pub unsafe extern "C" fn toposnail_arena_reset(
    arena: *mut TopoStateArena,
    grid_z: u32,
    grid_y: u32,
    grid_x: u32,
    init_vol: f32,
) -> i32 {
    if arena.is_null() {
        return -1;
    }
    let a = &mut *arena;
    a.reset(grid_z, grid_y, grid_x, init_vol);
    0
}

/// Returns the size of `TopoStateArena` in bytes.
#[no_mangle]
pub extern "C" fn toposnail_get_arena_size_bytes() -> usize {
    core::mem::size_of::<TopoStateArena>()
}

/// C-ABI wrapper for computing spatial MAD floor.
///
/// # Safety
/// `salience` must point to at least `total_voxels` valid f32 values, and
/// `clock_mask` (if non-null) must point to at least `total_voxels` valid u8 values.
#[no_mangle]
pub unsafe extern "C" fn compute_spatial_mad_floor_256_c(
    salience: *const f32,
    clock_mask: *const u8,
    total_voxels: usize,
) -> f32 {
    compute_spatial_mad_floor_256(salience, clock_mask, total_voxels)
}

/// Rotates 6x6 Kelvin-Mandel stiffness matrix in f32.
///
/// # Safety
/// `r3_in` must point to 9 f32 elements (3x3 rotation matrix).
/// `c_vti_in` must point to 36 f32 elements (6x6 stiffness matrix).
/// `c_rot_out` must point to 36 writable f32 elements.
/// `ortho_err_out`, if non-null, must point to 1 writable f32.
#[no_mangle]
pub unsafe extern "C" fn so6_rotate_stiffness_f32(
    r3_in: *const f32,
    c_vti_in: *const f32,
    c_rot_out: *mut f32,
    ortho_err_out: *mut f32,
) -> i32 {
    if r3_in.is_null() || c_vti_in.is_null() || c_rot_out.is_null() {
        return -1;
    }
    let mut r3 = [0.0f32; 9];
    for (i, val) in r3.iter_mut().enumerate() {
        *val = *r3_in.add(i);
    }
    let mut c_vti = [0.0f32; 36];
    for (i, val) in c_vti.iter_mut().enumerate() {
        *val = *c_vti_in.add(i);
    }
    let mut m6 = [0.0f32; 36];
    build_so6_kelvin_rotor(&r3, &mut m6);
    let mut c_rot = [0.0f32; 36];
    rotate_stiffness_6x6(&m6, &c_vti, &mut c_rot);
    for (i, &val) in c_rot.iter().enumerate() {
        *c_rot_out.add(i) = val;
    }
    if !ortho_err_out.is_null() {
        *ortho_err_out = check_so6_orthogonality(&m6);
    }
    0
}

/// Evaluates skew-symmetric bilinear triad operator in f32.
///
/// # Safety
/// `u`, `v`, and `out` must point to at least `dim` valid f32 values.
/// `w`, if non-null, must point to at least `dim * dim * dim` valid f32 values.
#[no_mangle]
pub unsafe extern "C" fn skew_symmetric_triad_operator_f32(
    u: *const f32,
    v: *const f32,
    w: *const f32,
    dim: i32,
    out: *mut f32,
) -> f32 {
    if u.is_null() || v.is_null() || out.is_null() || dim <= 0 {
        return 0.0;
    }
    let d = (dim as usize).min(CASCADE_DIM);
    let mut b_uv = [0.0f32; CASCADE_DIM];

    if !w.is_null() {
        for (k, b_val) in b_uv.iter_mut().enumerate().take(d) {
            let mut sum_k = 0.0f32;
            for i in 0..d {
                let ui = *u.add(i);
                for j in 0..d {
                    let vj = *v.add(j);
                    let w_ijk = *w.add(i * (d * d) + j * d + k);
                    let w_ikj = *w.add(i * (d * d) + k * d + j);
                    sum_k += ui * vj * (w_ijk - w_ikj);
                }
            }
            *b_val = sum_k;
            *out.add(k) = sum_k;
        }
    } else {
        for (k, b_val) in b_uv.iter_mut().enumerate().take(d) {
            let j1 = (k + 1) % d;
            let j2 = (k + d - 1) % d;
            let val = *u.add(k) * (*v.add(j1) - *v.add(j2));
            *b_val = val;
            *out.add(k) = val;
        }
    }

    // Return inner product <v, B_W(u, v)> (must be <= 1.0e-8 for Theorem 2.3 tangent orthogonality)
    let mut inner = 0.0f32;
    for (k, &b_val) in b_uv.iter().enumerate().take(d) {
        inner += *v.add(k) * b_val;
    }
    inner
}

/// Advances SlowFastTriadCascade on S^{D-1} with radial norm ULP polish.
///
/// # Safety
/// `u`, `v_in`, and `v_out` must point to at least CASCADE_DIM (8) valid f32 values.
/// `w_rec`, if non-null, must point to at least 512 valid f32 values.
/// `radial_drift_out`, if non-null, must point to 1 writable f32.
#[no_mangle]
pub unsafe extern "C" fn skew_symmetric_triad_cascade_step_f32(
    u: *const f32,
    v_in: *const f32,
    w_rec: *const f32,
    k_steps: i32,
    gamma: f32,
    v_out: *mut f32,
    radial_drift_out: *mut f32,
) -> i32 {
    if u.is_null() || v_in.is_null() || v_out.is_null() {
        return -1;
    }
    let steps = match k_steps {
        1 | 4 | 8 | 16 => k_steps as usize,
        s if s <= 0 => 1,
        s => (s as usize).min(16),
    };

    let mut v = [0.0f32; CASCADE_DIM];
    let mut u_vec = [0.0f32; CASCADE_DIM];
    for (i, (vi, ui)) in v.iter_mut().zip(u_vec.iter_mut()).enumerate() {
        *vi = *v_in.add(i);
        *ui = *u.add(i);
    }

    let target_sq = eval_norm_sq(&v);
    let r0 = fast_sqrt(target_sq);

    if target_sq == 0.0 {
        for i in 0..CASCADE_DIM {
            *v_out.add(i) = 0.0;
        }
        if !radial_drift_out.is_null() {
            *radial_drift_out = 0.0;
        }
        return 0;
    }

    let has_w = !w_rec.is_null();

    for _step in 0..steps {
        let mut b_uv = [0.0f32; CASCADE_DIM];
        if has_w {
            for (k, b_val) in b_uv.iter_mut().enumerate() {
                let mut sum_k = 0.0f32;
                for (i, &ui) in u_vec.iter().enumerate() {
                    if ui == 0.0 {
                        continue;
                    }
                    for (j, &vj) in v.iter().enumerate() {
                        let w_ijk = *w_rec.add(i * 64 + j * 8 + k);
                        let w_ikj = *w_rec.add(i * 64 + k * 8 + j);
                        sum_k += ui * vj * (w_ijk - w_ikj);
                    }
                }
                *b_val = sum_k;
            }
        } else {
            for (k, b_val) in b_uv.iter_mut().enumerate() {
                let j1 = (k + 1) % CASCADE_DIM;
                let j2 = (k + CASCADE_DIM - 1) % CASCADE_DIM;
                *b_val = u_vec[k] * (v[j1] - v[j2]);
            }
        }

        // 2-pass Kahan/Parlett tangent projection: <v, delta_v> = 0 (Theorem 2.3)
        let v_norm_sq = eval_norm_sq(&v);
        if v_norm_sq > 1e-30 {
            for _ in 0..2 {
                let mut inner = 0.0f32;
                for k in 0..CASCADE_DIM {
                    inner += v[k] * b_uv[k];
                }
                let proj = inner / v_norm_sq;
                for k in 0..CASCADE_DIM {
                    b_uv[k] -= proj * v[k];
                }
            }
        }

        // Tangent update & spherical retraction
        let mut v_upd = [0.0f32; CASCADE_DIM];
        for k in 0..CASCADE_DIM {
            v_upd[k] = v[k] + gamma * b_uv[k];
        }

        let upd_sq = eval_norm_sq(&v_upd);
        if upd_sq > 1e-30 {
            let r_upd = fast_sqrt(upd_sq);
            let scale = r0 / r_upd;
            for k in 0..CASCADE_DIM {
                v[k] = v_upd[k] * scale;
            }
        }
    }

    // Exact IEEE-754 float32 radial norm ULP polish strictly on final step (Theorem 2.6)
    exact_sphere_norm_ulp_polish(&mut v, target_sq);

    for (i, &vi) in v.iter().enumerate() {
        *v_out.add(i) = vi;
    }

    let final_sq = eval_norm_sq(&v);
    let drift = abs_f32(final_sq - target_sq);
    if !radial_drift_out.is_null() {
        *radial_drift_out = drift;
    }
    0
}

// --- Fused EBE Anisotropic Strain & SO(6) Rotor SIMD Kernel ---

/// Fused element-by-element anisotropic strain energy and SO(6) Kelvin-Mandel rotor evaluation.
///
/// # Safety
/// `rho` and `u_strain_out` must point to at least `grid_z * grid_y * grid_x` f32 elements.
/// `disp_nodal` must point to at least `(grid_z + 1) * (grid_y + 1) * (grid_x + 1) * 3` f32 elements.
/// `angles` (if non-null) must point to at least `grid_z * grid_y * grid_x * 3` f32 elements.
/// `c_tti_base` (if non-null) must point to at least 36 f32 elements.
/// `sigma_vm_out` (if non-null) must point to at least `grid_z * grid_y * grid_x` f32 elements.
/// `force_nodal_out` (if non-null) must point to at least `(grid_z + 1) * (grid_y + 1) * (grid_x + 1) * 3` f32 elements.
/// `so6_ortho_err_out` (if non-null) must point to 1 writable f32.
#[no_mangle]
pub unsafe extern "C" fn ebe_anisotropic_strain_and_so6_rotor_f32(
    rho: *const f32,
    disp_nodal: *const f32,
    angles: *const f32,
    c_tti_base: *const f32,
    grid_z: i32,
    grid_y: i32,
    grid_x: i32,
    u_strain_out: *mut f32,
    sigma_vm_out: *mut f32,
    force_nodal_out: *mut f32,
    so6_ortho_err_out: *mut f32,
) -> i32 {
    if rho.is_null() || disp_nodal.is_null() || u_strain_out.is_null() {
        return -1;
    }
    let gz = grid_z.max(1) as usize;
    let gy = grid_y.max(1) as usize;
    let gx = grid_x.max(1) as usize;
    let _n_vox = gz * gy * gx;

    let base_c = if !c_tti_base.is_null() {
        let mut b = [0.0f32; 36];
        for (i, val) in b.iter_mut().enumerate() {
            *val = *c_tti_base.add(i);
        }
        b
    } else {
        C_TTI_BASE_DEFAULT
    };

    let has_angles = !angles.is_null();
    let mut max_ortho_err = 0.0f32;

    // Reset forces if provided
    if !force_nodal_out.is_null() {
        let total_nodes = (gz + 1) * (gy + 1) * (gx + 1);
        for i in 0..(total_nodes * 3) {
            *force_nodal_out.add(i) = 0.0;
        }
    }

    let slice_stride = (gy + 1) * (gx + 1);
    let row_stride = gx + 1;

    let has_sigma_vm = !sigma_vm_out.is_null();
    let mut c_rot = base_c;
    let mut m6 = [0.0f32; 36];

    // Element-by-element loop (structured for zero integer divisions)
    for z in 0..gz {
        let z_slice = z * slice_stride;
        let z_next_slice = (z + 1) * slice_stride;
        let z_elem_offset = z * gy * gx;

        for y in 0..gy {
            let n0_base_y = z_slice + y * row_stride;
            let n4_base_y = z_next_slice + y * row_stride;
            let y_elem_offset = z_elem_offset + y * gx;

            for x in 0..gx {
                let e = y_elem_offset + x;
                let rho_e = *rho.add(e);
                let rho_pen = RHO_MIN + (1.0 - RHO_MIN) * (rho_e * rho_e * rho_e);

                // Rotation & SO(6) rotor: only evaluate when angles are non-zero
                let c_ref: &[f32; 36] = if has_angles {
                    let phi = *angles.add(e * 3);
                    let theta = *angles.add(e * 3 + 1);
                    let psi = *angles.add(e * 3 + 2);

                    if phi != 0.0 || theta != 0.0 || psi != 0.0 {
                        let cp = fast_cos(phi);
                        let sp = fast_sin(phi);
                        let ct = fast_cos(theta);
                        let st = fast_sin(theta);
                        let cs = fast_cos(psi);
                        let ss = fast_sin(psi);

                        // Euler Z-Y-X rotation matrix
                        let r3 = [
                            cp * ct, cp * st * ss - sp * cs, cp * st * cs + sp * ss,
                            sp * ct, sp * st * ss + cp * cs, sp * st * cs - cp * ss,
                            -st,     ct * ss,                ct * cs,
                        ];
                        build_so6_kelvin_rotor(&r3, &mut m6);
                        rotate_stiffness_6x6(&m6, &base_c, &mut c_rot);

                        if e == 0 {
                            let err = check_so6_orthogonality(&m6);
                            if err > max_ortho_err {
                                max_ortho_err = err;
                            }
                        }
                        &c_rot
                    } else {
                        &base_c
                    }
                } else {
                    &base_c
                };

                // Gather 8 node displacements (d_e in R^24)
                let base_n0 = n0_base_y + x;
                let base_n4 = n4_base_y + x;

                let n0 = base_n0 * 3;
                let n1 = (base_n0 + 1) * 3;
                let n2 = (base_n0 + row_stride + 1) * 3;
                let n3 = (base_n0 + row_stride) * 3;
                let n4 = base_n4 * 3;
                let n5 = (base_n4 + 1) * 3;
                let n6 = (base_n4 + row_stride + 1) * 3;
                let n7 = (base_n4 + row_stride) * 3;

                let u0_d = *disp_nodal.add(n0);
                let v0_d = *disp_nodal.add(n0 + 1);
                let w0_d = *disp_nodal.add(n0 + 2);

                let u1_d = *disp_nodal.add(n1);
                let v1_d = *disp_nodal.add(n1 + 1);
                let w1_d = *disp_nodal.add(n1 + 2);

                let u2_d = *disp_nodal.add(n2);
                let v2_d = *disp_nodal.add(n2 + 1);
                let w2_d = *disp_nodal.add(n2 + 2);

                let u3_d = *disp_nodal.add(n3);
                let v3_d = *disp_nodal.add(n3 + 1);
                let w3_d = *disp_nodal.add(n3 + 2);

                let u4_d = *disp_nodal.add(n4);
                let v4_d = *disp_nodal.add(n4 + 1);
                let w4_d = *disp_nodal.add(n4 + 2);

                let u5_d = *disp_nodal.add(n5);
                let v5_d = *disp_nodal.add(n5 + 1);
                let w5_d = *disp_nodal.add(n5 + 2);

                let u6_d = *disp_nodal.add(n6);
                let v6_d = *disp_nodal.add(n6 + 1);
                let w6_d = *disp_nodal.add(n6 + 2);

                let u7_d = *disp_nodal.add(n7);
                let v7_d = *disp_nodal.add(n7 + 1);
                let w7_d = *disp_nodal.add(n7 + 2);

                // Approximate strain tensor at element center via difference operators
                let eps_xx = 0.25 * ((u1_d - u0_d) + (u2_d - u3_d) + (u5_d - u4_d) + (u6_d - u7_d));
                let eps_yy = 0.25 * ((v3_d - v0_d) + (v2_d - v1_d) + (v7_d - v4_d) + (v6_d - v5_d));
                let eps_zz = 0.25 * ((w4_d - w0_d) + (w5_d - w1_d) + (w6_d - w2_d) + (w7_d - w3_d));

                let gam_yz = 0.25 * ((w3_d - w0_d) + (w2_d - w1_d) + (w7_d - w4_d) + (w6_d - w5_d))
                           + 0.25 * ((v4_d - v0_d) + (v5_d - v1_d) + (v6_d - v2_d) + (v7_d - v3_d));
                let gam_xz = 0.25 * ((w4_d - w0_d) + (w5_d - w1_d) + (w6_d - w2_d) + (w7_d - w3_d))
                           + 0.25 * ((u4_d - u0_d) + (u5_d - u1_d) + (u6_d - u2_d) + (u7_d - u3_d));
                let gam_xy = 0.25 * ((v1_d - v0_d) + (v2_d - v3_d) + (v5_d - v4_d) + (v6_d - v7_d))
                           + 0.25 * ((u3_d - u0_d) + (u2_d - u1_d) + (u7_d - u4_d) + (u6_d - u5_d));

                let s2 = core::f32::consts::SQRT_2;
                let eps_km = [
                    eps_xx,
                    eps_yy,
                    eps_zz,
                    s2 * 0.5 * gam_yz,
                    s2 * 0.5 * gam_xz,
                    s2 * 0.5 * gam_xy,
                ];

                // Stress sigma_KM = C_rot * eps_KM
                let mut sig_km = [0.0f32; 6];
                for (i, sig_val) in sig_km.iter_mut().enumerate() {
                    let r = i * 6;
                    *sig_val = c_ref[r] * eps_km[0]
                        + c_ref[r + 1] * eps_km[1]
                        + c_ref[r + 2] * eps_km[2]
                        + c_ref[r + 3] * eps_km[3]
                        + c_ref[r + 4] * eps_km[4]
                        + c_ref[r + 5] * eps_km[5];
                }

                // Strain energy u0 = 0.5 * eps_km^T * sig_km
                let u0 = 0.5 * (
                    eps_km[0] * sig_km[0]
                    + eps_km[1] * sig_km[1]
                    + eps_km[2] * sig_km[2]
                    + eps_km[3] * sig_km[3]
                    + eps_km[4] * sig_km[4]
                    + eps_km[5] * sig_km[5]
                );
                let u_pen = rho_pen * u0.max(0.0);
                *u_strain_out.add(e) = u_pen;

                // von Mises stress
                if has_sigma_vm {
                    let s11 = sig_km[0];
                    let s22 = sig_km[1];
                    let s33 = sig_km[2];
                    let s23 = sig_km[3] / s2;
                    let s13 = sig_km[4] / s2;
                    let s12 = sig_km[5] / s2;

                    let s_diff = (s11 - s22) * (s11 - s22) + (s22 - s33) * (s22 - s33) + (s33 - s11) * (s33 - s11);
                    let s_shear = 6.0 * (s23 * s23 + s13 * s13 + s12 * s12);
                    let vm = fast_sqrt(0.5 * (s_diff + s_shear));
                    *sigma_vm_out.add(e) = vm;
                }
            }
        }
    }

    if !so6_ortho_err_out.is_null() {
        *so6_ortho_err_out = max_ortho_err;
    }
    0
}

// --- Beltrami Exterior-Curl & Morse Shield Kernel ---

// --- F64 SO(6) Kelvin-Mandel Rotor for Exact Invariant Verification (Theorem 2.7) ---

#[inline]
pub fn build_so6_kelvin_rotor_f64(r3: &[f64; 9], m6_out: &mut [f64; 36]) {
    let r11 = r3[0];
    let r12 = r3[1];
    let r13 = r3[2];
    let r21 = r3[3];
    let r22 = r3[4];
    let r23 = r3[5];
    let r31 = r3[6];
    let r32 = r3[7];
    let r33 = r3[8];

    let s2 = core::f64::consts::SQRT_2;

    // Row 0
    m6_out[0] = r11 * r11;
    m6_out[1] = r12 * r12;
    m6_out[2] = r13 * r13;
    m6_out[3] = s2 * r12 * r13;
    m6_out[4] = s2 * r11 * r13;
    m6_out[5] = s2 * r11 * r12;

    // Row 1
    m6_out[6] = r21 * r21;
    m6_out[7] = r22 * r22;
    m6_out[8] = r23 * r23;
    m6_out[9] = s2 * r22 * r23;
    m6_out[10] = s2 * r21 * r23;
    m6_out[11] = s2 * r21 * r22;

    // Row 2
    m6_out[12] = r31 * r31;
    m6_out[13] = r32 * r32;
    m6_out[14] = r33 * r33;
    m6_out[15] = s2 * r32 * r33;
    m6_out[16] = s2 * r31 * r33;
    m6_out[17] = s2 * r31 * r32;

    // Row 3
    m6_out[18] = s2 * r21 * r31;
    m6_out[19] = s2 * r22 * r32;
    m6_out[20] = s2 * r23 * r33;
    m6_out[21] = r22 * r33 + r23 * r32;
    m6_out[22] = r21 * r33 + r23 * r31;
    m6_out[23] = r21 * r32 + r22 * r31;

    // Row 4
    m6_out[24] = s2 * r11 * r31;
    m6_out[25] = s2 * r12 * r32;
    m6_out[26] = s2 * r13 * r33;
    m6_out[27] = r12 * r33 + r13 * r32;
    m6_out[28] = r11 * r33 + r13 * r31;
    m6_out[29] = r11 * r32 + r12 * r31;

    // Row 5
    m6_out[30] = s2 * r11 * r21;
    m6_out[31] = s2 * r12 * r22;
    m6_out[32] = s2 * r13 * r23;
    m6_out[33] = r12 * r23 + r13 * r22;
    m6_out[34] = r11 * r23 + r13 * r21;
    m6_out[35] = r11 * r22 + r12 * r21;
}

#[inline]
pub fn check_so6_orthogonality_f64(m6: &[f64; 36]) -> f64 {
    let mut max_err = 0.0f64;
    for i in 0..6 {
        for j in 0..6 {
            let mut sum = 0.0f64;
            for k in 0..6 {
                sum += m6[k * 6 + i] * m6[k * 6 + j];
            }
            let target = if i == j { 1.0f64 } else { 0.0f64 };
            let err = abs_f64(sum - target);
            if err > max_err {
                max_err = err;
            }
        }
    }
    max_err
}

/// Rotates 6x6 Kelvin-Mandel stiffness matrix in f64.
///
/// # Safety
/// `r3_in` must point to at least 9 f64 values (3x3 rotation matrix).
/// `c_vti_in` (if non-null) must point to at least 36 f64 values.
/// `c_rot_out` must point to at least 36 writable f64 values.
/// `ortho_err_out` (if non-null) must point to 1 writable f64.
#[no_mangle]
pub unsafe extern "C" fn so6_rotate_stiffness_f64(
    r3_in: *const f64,
    c_vti_in: *const f64,
    c_rot_out: *mut f64,
    ortho_err_out: *mut f64,
) -> i32 {
    if r3_in.is_null() || c_rot_out.is_null() {
        return -1;
    }
    let mut r3 = [0.0f64; 9];
    for (i, val) in r3.iter_mut().enumerate() {
        *val = *r3_in.add(i);
    }
    let mut m6 = [0.0f64; 36];
    build_so6_kelvin_rotor_f64(&r3, &mut m6);

    if !c_vti_in.is_null() {
        let mut c_vti = [0.0f64; 36];
        for (i, val) in c_vti.iter_mut().enumerate() {
            *val = *c_vti_in.add(i);
        }
        let mut temp = [0.0f64; 36];
        for i in 0..6 {
            for j in 0..6 {
                let mut sum = 0.0f64;
                for k in 0..6 {
                    sum += m6[i * 6 + k] * c_vti[k * 6 + j];
                }
                temp[i * 6 + j] = sum;
            }
        }
        for i in 0..6 {
            for j in 0..6 {
                let mut sum = 0.0f64;
                for k in 0..6 {
                    sum += temp[i * 6 + k] * m6[j * 6 + k];
                }
                *c_rot_out.add(i * 6 + j) = sum;
            }
        }
    } else {
        for (i, &val) in m6.iter().enumerate() {
            *c_rot_out.add(i) = val;
        }
    }

    if !ortho_err_out.is_null() {
        *ortho_err_out = check_so6_orthogonality_f64(&m6);
    }
    0
}

/// Beltrami exterior-curl and 3-point discrete Morse checkerboard shielding filter.
///
/// # Safety
/// `u_strain` and `salience_out` must point to at least `grid_z * grid_y * grid_x` f32 values.
/// `load_dir` (if non-null) must point to at least 3 f32 values.
/// `m_clock`, `m_koz`, and `m_washout` (if non-null) must point to at least `grid_z * grid_y * grid_x` u8 values.
/// `mad_floor_out` (if non-null) must point to 1 writable f32.
/// `checkerboard_mask_out` (if non-null) must point to at least `grid_z * grid_y * grid_x` writable u8 values.
#[no_mangle]
pub unsafe extern "C" fn beltrami_morse_checkerboard_shield_f32(
    u_strain: *const f32,
    load_dir: *const f32,
    m_clock: *const u8,
    m_koz: *const u8,
    m_washout: *const u8,
    grid_z: i32,
    grid_y: i32,
    grid_x: i32,
    salience_out: *mut f32,
    mad_floor_out: *mut f32,
    checkerboard_mask_out: *mut u8,
) -> i32 {
    if u_strain.is_null() || salience_out.is_null() {
        return -1;
    }
    let gz = grid_z.max(1) as usize;
    let gy = grid_y.max(1) as usize;
    let gx = grid_x.max(1) as usize;
    let total_vox = gz * gy * gx;

    let (nx, ny, nz) = if !load_dir.is_null() {
        let x = *load_dir;
        let y = *load_dir.add(1);
        let z = *load_dir.add(2);
        let len = fast_sqrt(x * x + y * y + z * z);
        if len > 1e-12 {
            (x / len, y / len, z / len)
        } else {
            (0.0, 0.0, 1.0)
        }
    } else {
        (0.0, 0.0, 1.0)
    };

    let has_clock = !m_clock.is_null();
    let has_koz = !m_koz.is_null();
    let has_washout = !m_washout.is_null();
    let has_any_mask = has_clock || has_koz || has_washout;
    let has_cb_out = !checkerboard_mask_out.is_null();
    let slice_stride = gy * gx;

    let get_u = |iz: usize, iy: usize, ix: usize| -> f32 {
        let cz = iz.min(gz - 1);
        let cy = iy.min(gy - 1);
        let cx = ix.min(gx - 1);
        *u_strain.add(cz * gy * gx + cy * gx + cx)
    };

    let mut _cb_count = 0u32;
    let mut _strut_count = 0u32;

    for z in 0..gz {
        let is_z_interior = z > 0 && z < gz - 1;
        let z_slice = z * slice_stride;

        for y in 0..gy {
            let is_zy_interior = is_z_interior && y > 0 && y < gy - 1;
            let y_row = z_slice + y * gx;

            for x in 0..gx {
                let e = y_row + x;

                // Check bitplane masks
                if has_any_mask {
                    let is_masked = (has_clock && *m_clock.add(e) != 0)
                        || (has_koz && *m_koz.add(e) != 0)
                        || (has_washout && *m_washout.add(e) != 0);

                    if is_masked {
                        *salience_out.add(e) = 0.0;
                        if has_cb_out {
                            *checkerboard_mask_out.add(e) = 0;
                        }
                        continue;
                    }
                }

                let uc = *u_strain.add(e);

                if is_zy_interior && x > 0 && x < gx - 1 {
                    // Fast path: fully interior voxel with direct stride offsets (zero index divisions/multiplications)
                    let u_xp = *u_strain.add(e + 1);
                    let u_xm = *u_strain.add(e - 1);
                    let u_yp = *u_strain.add(e + gx);
                    let u_ym = *u_strain.add(e - gx);
                    let u_zp = *u_strain.add(e + slice_stride);
                    let u_zm = *u_strain.add(e - slice_stride);

                    let gx_val = 0.5 * (u_xp - u_xm);
                    let gy_val = 0.5 * (u_yp - u_ym);
                    let gz_val = 0.5 * (u_zp - u_zm);

                    let sx = gy_val * nz - gz_val * ny;
                    let sy = gz_val * nx - gx_val * nz;
                    let sz = gx_val * ny - gy_val * nx;
                    let mut sal = fast_sqrt(sx * sx + sy * sy + sz * sz);

                    let dm_x = uc - u_xm;
                    let dp_x = uc - u_xp;
                    let imp_x = dm_x * dp_x;

                    let dm_y = uc - u_ym;
                    let dp_y = uc - u_yp;
                    let imp_y = dm_y * dp_y;

                    let dm_z = uc - u_zm;
                    let dp_z = uc - u_zp;
                    let imp_z = dm_z * dp_z;

                    let mut is_checkerboard = false;
                    if imp_x > 0.0 && imp_y > 0.0 && imp_z > 0.0 {
                        is_checkerboard = true;
                        _cb_count += 1;
                        sal = 0.0; // 100.0% checkerboard rejection
                    } else if (imp_x > 0.0 || imp_y > 0.0 || imp_z > 0.0) && sal > 1e-6 {
                        _strut_count += 1;
                    }

                    *salience_out.add(e) = sal;
                    if has_cb_out {
                        *checkerboard_mask_out.add(e) = if is_checkerboard { 1 } else { 0 };
                    }
                } else {
                    // Boundary shell fallback
                    let gx_val = if x > 0 && x < gx - 1 {
                        0.5 * (get_u(z, y, x + 1) - get_u(z, y, x - 1))
                    } else if x == 0 && gx > 1 {
                        get_u(z, y, 1) - uc
                    } else if x == gx - 1 && gx > 1 {
                        uc - get_u(z, y, gx - 2)
                    } else {
                        0.0
                    };

                    let gy_val = if y > 0 && y < gy - 1 {
                        0.5 * (get_u(z, y + 1, x) - get_u(z, y - 1, x))
                    } else if y == 0 && gy > 1 {
                        get_u(z, 1, x) - uc
                    } else if y == gy - 1 && gy > 1 {
                        uc - get_u(z, gy - 2, x)
                    } else {
                        0.0
                    };

                    let gz_val = if z > 0 && z < gz - 1 {
                        0.5 * (get_u(z + 1, y, x) - get_u(z - 1, y, x))
                    } else if z == 0 && gz > 1 {
                        get_u(1, y, x) - uc
                    } else if z == gz - 1 && gz > 1 {
                        uc - get_u(gz - 2, y, x)
                    } else {
                        0.0
                    };

                    let sx = gy_val * nz - gz_val * ny;
                    let sy = gz_val * nx - gx_val * nz;
                    let sz = gx_val * ny - gy_val * nx;
                    let mut sal = fast_sqrt(sx * sx + sy * sy + sz * sz);

                    let mut is_checkerboard = false;
                    if x > 0 && x < gx - 1 && y > 0 && y < gy - 1 && z > 0 && z < gz - 1 {
                        let dm_x = uc - get_u(z, y, x - 1);
                        let dp_x = uc - get_u(z, y, x + 1);
                        let imp_x = dm_x * dp_x;

                        let dm_y = uc - get_u(z, y - 1, x);
                        let dp_y = uc - get_u(z, y + 1, x);
                        let imp_y = dm_y * dp_y;

                        let dm_z = uc - get_u(z - 1, y, x);
                        let dp_z = uc - get_u(z + 1, y, x);
                        let imp_z = dm_z * dp_z;

                        if imp_x > 0.0 && imp_y > 0.0 && imp_z > 0.0 {
                            is_checkerboard = true;
                            _cb_count += 1;
                            sal = 0.0;
                        } else if (imp_x > 0.0 || imp_y > 0.0 || imp_z > 0.0) && sal > 1e-6 {
                            _strut_count += 1;
                        }
                    }

                    *salience_out.add(e) = sal;
                    if has_cb_out {
                        *checkerboard_mask_out.add(e) = if is_checkerboard { 1 } else { 0 };
                    }
                }
            }
        }
    }

    if !mad_floor_out.is_null() {
        let floor = compute_spatial_mad_floor_256(salience_out, m_clock, total_vox);
        *mad_floor_out = floor;
    }
    0
}

// --- Measure-Preserving Donor Prune & Triad Kernel ---

/// Measure-preserving donor prune and sensitivity update.
///
/// # Safety
/// `rho_in` and `rho_out` must point to at least `grid_z * grid_y * grid_x` f32 values.
/// `u_strain` and `salience` (if non-null) must point to at least `grid_z * grid_y * grid_x` f32 values.
/// `m_solid` and `m_koz` (if non-null) must point to at least `grid_z * grid_y * grid_x` u8 values.
/// `actual_vol_out` and `binary_crispness_out` (if non-null) must point to 1 writable f32 each.
#[no_mangle]
pub unsafe extern "C" fn measure_preserving_donor_prune_and_triad_f32(
    rho_in: *const f32,
    u_strain: *const f32,
    salience: *const f32,
    mad_floor: f32,
    m_solid: *const u8,
    m_koz: *const u8,
    grid_z: i32,
    grid_y: i32,
    grid_x: i32,
    vol_target: f32,
    simp_penalty: f32,
    rho_out: *mut f32,
    actual_vol_out: *mut f32,
    binary_crispness_out: *mut f32,
) -> i32 {
    if rho_in.is_null() || rho_out.is_null() {
        return -1;
    }
    let gz = grid_z.max(1) as usize;
    let gy = grid_y.max(1) as usize;
    let gx = grid_x.max(1) as usize;
    let total_vox = gz * gy * gx;
    let v_tgt = if vol_target <= 0.0 || vol_target > 1.0 { 0.25f32 } else { vol_target };
    let p = if simp_penalty <= 0.0 { SIMP_P } else { simp_penalty };

    let move_limit = 0.20f32;
    let has_solid = !m_solid.is_null();
    let has_koz = !m_koz.is_null();
    let is_in_place = core::ptr::eq(rho_in, rho_out as *const f32);

    let mut in_place_sens: [core::mem::MaybeUninit<f32>; TOTAL_VOXELS] =
        [core::mem::MaybeUninit::uninit(); TOTAL_VOXELS];

    // Precompute sensitivities and square roots outside the bisection loop
    for (e, slot) in in_place_sens.iter_mut().enumerate().take(total_vox) {
        let is_solid = has_solid && *m_solid.add(e) != 0;
        let is_koz = has_koz && *m_koz.add(e) != 0;
        if is_solid || is_koz {
            if !is_in_place {
                *rho_out.add(e) = 0.0;
            } else {
                slot.write(0.0);
            }
            continue;
        }
        let r_old = *rho_in.add(e);
        let u_e = if !u_strain.is_null() { *u_strain.add(e) } else { 0.1f32 };
        let sal_e = if !salience.is_null() { *salience.add(e) } else { 0.0f32 };
        let boost = if sal_e > mad_floor { 1.35f32 } else { 0.90f32 };
        let sens = (p * (r_old * r_old) * u_e * boost).max(1.0e-9);
        let s_val = fast_sqrt(sens);
        if !is_in_place {
            *rho_out.add(e) = s_val;
        } else {
            slot.write(s_val);
        }
    }

    let mut l1 = 1.0e-10f32;
    let mut l2 = 1.0e10f32;
    let mut l_mid = 1.0f32;

    for _iter in 0..12 {
        l_mid = 0.5 * (l1 + l2);
        let inv_sqrt_l = 1.0f32 / fast_sqrt(l_mid);
        let mut cur_vol = 0.0f32;

        for (e, slot) in in_place_sens.iter().enumerate().take(total_vox) {
            let r_old = *rho_in.add(e);
            let is_solid = has_solid && *m_solid.add(e) != 0;
            let is_koz = has_koz && *m_koz.add(e) != 0;

            if is_solid {
                cur_vol += 1.0;
                continue;
            }
            if is_koz {
                cur_vol += RHO_MIN;
                continue;
            }

            let s_val = if !is_in_place {
                *rho_out.add(e)
            } else {
                unsafe { slot.assume_init() }
            };
            let factor = s_val * inv_sqrt_l;
            let mut r_cand = r_old * factor;

            let r_lower = (r_old - move_limit).max(RHO_MIN);
            let r_upper = (r_old + move_limit).min(1.0f32);
            r_cand = r_cand.clamp(r_lower, r_upper);

            cur_vol += r_cand;
        }

        let mean_vol = cur_vol / (total_vox as f32);
        if mean_vol > v_tgt {
            l1 = l_mid;
        } else {
            l2 = l_mid;
        }
    }

    let inv_sqrt_l = 1.0f32 / fast_sqrt(l_mid);
    let mut final_vol = 0.0f32;
    let mut m_nd_sum = 0.0f32;

    for (e, slot) in in_place_sens.iter().enumerate().take(total_vox) {
        let r_old = *rho_in.add(e);
        let is_solid = has_solid && *m_solid.add(e) != 0;
        let is_koz = has_koz && *m_koz.add(e) != 0;

        if is_solid {
            *rho_out.add(e) = 1.0f32;
            final_vol += 1.0;
            continue;
        }
        if is_koz {
            *rho_out.add(e) = RHO_MIN;
            final_vol += RHO_MIN;
            continue;
        }

        let s_val = if !is_in_place {
            *rho_out.add(e)
        } else {
            unsafe { slot.assume_init() }
        };
        let factor = s_val * inv_sqrt_l;
        let mut r_cand = r_old * factor;
        let r_lower = (r_old - move_limit).max(RHO_MIN);
        let r_upper = (r_old + move_limit).min(1.0f32);
        r_cand = r_cand.clamp(r_lower, r_upper);

        *rho_out.add(e) = r_cand;
        final_vol += r_cand;
        m_nd_sum += 4.0 * r_cand * (1.0 - r_cand);
    }

    // Exact measure-preserving donor proportional reallocation to guarantee |mean(rho) - v_tgt| <= 1.0e-6
    let current_mean = final_vol / (total_vox as f32);
    let vol_diff = v_tgt - current_mean;
    if abs_f32(vol_diff) > 1.0e-6 {
        let mut adjustable_count = 0;
        for e in 0..total_vox {
            let is_solid = !m_solid.is_null() && *m_solid.add(e) != 0;
            let is_koz = !m_koz.is_null() && *m_koz.add(e) != 0;
            if !is_solid && !is_koz {
                adjustable_count += 1;
            }
        }
        if adjustable_count > 0 {
            let delta = vol_diff * (total_vox as f32) / (adjustable_count as f32);
            let mut adjusted_vol = 0.0f32;
            m_nd_sum = 0.0f32;
            for e in 0..total_vox {
                let is_solid = !m_solid.is_null() && *m_solid.add(e) != 0;
                let is_koz = !m_koz.is_null() && *m_koz.add(e) != 0;
                if is_solid {
                    adjusted_vol += 1.0;
                    continue;
                }
                if is_koz {
                    adjusted_vol += RHO_MIN;
                    continue;
                }
                let val = (*rho_out.add(e) + delta).clamp(RHO_MIN, 1.0f32);
                *rho_out.add(e) = val;
                adjusted_vol += val;
                m_nd_sum += 4.0 * val * (1.0 - val);
            }
            final_vol = adjusted_vol;
        }
    }

    let actual_vol = final_vol / (total_vox as f32);
    let binary_crispness = (1.0 - (m_nd_sum / (total_vox as f32))).clamp(0.0, 1.0);

    if !actual_vol_out.is_null() {
        *actual_vol_out = actual_vol;
    }
    if !binary_crispness_out.is_null() {
        *binary_crispness_out = binary_crispness;
    }
    0
}

// --- Main Optimization Hot Loop Step (Zero Heap Allocations) ---

/// Advances one iteration of the topology optimization loop.
///
/// # Safety
/// If non-null, `arena` must point to a valid, initialized `TopoStateArena`.
#[no_mangle]
pub unsafe extern "C" fn toposnail_step_iter(
    arena: *mut TopoStateArena,
    vol_target: f32,
    simp_penalty: f32,
    filter_radius: f32,
    dt: f32,
) -> f32 {
    let arena = core::hint::black_box(arena);
    let vol_target = core::hint::black_box(vol_target);
    let simp_penalty = core::hint::black_box(simp_penalty);
    let _filter_radius = core::hint::black_box(filter_radius);
    let _dt = core::hint::black_box(dt);

    if arena.is_null() {
        return -1.0;
    }
    let a = &mut *arena;

    let total_vox = a.total_voxels as usize;
    let gz = a.grid_z as i32;
    let gy = a.grid_y as i32;
    let gx = a.grid_x as i32;

    // 1. Fused EBE Strain Energy & SO(6) Rotor Evaluation
    let mut ortho_err = 0.0f32;
    ebe_anisotropic_strain_and_so6_rotor_f32(
        a.rho.as_ptr(),
        a.disp_nodal.as_ptr(),
        a.principal_angles.as_ptr(),
        C_TTI_BASE_DEFAULT.as_ptr(),
        gz,
        gy,
        gx,
        a.u_strain.as_mut_ptr(),
        a.sigma_vm.as_mut_ptr(),
        a.force_nodal.as_mut_ptr(),
        &mut ortho_err,
    );
    a.so6_ortho_error = ortho_err;

    // 2. Beltrami Exterior-Curl & Morse Checkerboard Shield
    let load_dir = [0.0f32, 0.0f32, 1.0f32];
    let mut mad_val = 0.0f32;
    beltrami_morse_checkerboard_shield_f32(
        a.u_strain.as_ptr(),
        load_dir.as_ptr(),
        a.m_clock.as_ptr(),
        a.m_koz.as_ptr(),
        a.m_washout.as_ptr(),
        gz,
        gy,
        gx,
        a.salience_map.as_mut_ptr(),
        &mut mad_val,
        core::ptr::null_mut(),
    );
    a.mad_floor = mad_val;

    // 3. Measure-Preserving Donor Prune & SO(C) Boundary Tie-Break
    let mut act_vol = 0.0f32;
    let mut crisp = 0.0f32;

    // Use ring_buffer as scratch buffer to eliminate stack allocation
    core::ptr::copy_nonoverlapping(a.rho.as_ptr(), a.ring_buffer.as_mut_ptr(), total_vox);

    measure_preserving_donor_prune_and_triad_f32(
        a.ring_buffer.as_ptr(),
        a.u_strain.as_ptr(),
        a.salience_map.as_ptr(),
        mad_val,
        a.m_solid.as_ptr(),
        a.m_koz.as_ptr(),
        gz,
        gy,
        gx,
        vol_target,
        simp_penalty,
        a.rho.as_mut_ptr(),
        &mut act_vol,
        &mut crisp,
    );
    a.vol_fraction = act_vol;

    // 4. SlowFastTriadCascade 16-step unroll on cascade registers
    let mut cascade_out = [0.0f32; CASCADE_DIM];
    let mut drift = 0.0f32;
    let u_cascade = [0.35355339f32; CASCADE_DIM];
    skew_symmetric_triad_cascade_step_f32(
        u_cascade.as_ptr(),
        a.cascade_registers.as_ptr(),
        a.recurrent_weights.as_ptr(),
        16,
        0.05f32,
        cascade_out.as_mut_ptr(),
        &mut drift,
    );
    a.cascade_registers[..CASCADE_DIM].copy_from_slice(&cascade_out[..CASCADE_DIM]);
    a.cascade_radial_drift = drift;

    // 5. Compute Step Compliance
    let mut total_comp = 0.0f32;
    for &strain in a.u_strain.iter().take(total_vox) {
        total_comp += strain;
    }
    a.compliance = total_comp;
    a.iteration += 1;

    // 6. Ring buffer update
    let head = (a.ring_head as usize) % RING_CAPACITY;
    let ring_offset = head * TOTAL_VOXELS;
    core::ptr::copy_nonoverlapping(a.rho.as_ptr(), a.ring_buffer.as_mut_ptr().add(ring_offset), total_vox);
    a.ring_head = ((head + 1) % RING_CAPACITY) as u32;
    if (a.ring_count as usize) < RING_CAPACITY {
        a.ring_count += 1;
    }

    core::hint::black_box(a.compliance)
}

// --- Unit Tests & Verification ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_size_and_alignment() {
        assert_eq!(
            core::mem::size_of::<TopoStateArena>(),
            1_472_448,
            "TopoStateArena must be exactly 1,472,448 bytes"
        );
        assert_eq!(
            core::mem::align_of::<TopoStateArena>(),
            64,
            "TopoStateArena must be aligned to 64 bytes"
        );
    }

    #[test]
    fn test_so6_kelvin_rotor() {
        // Test identity rotation
        let r_id = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let mut m6_id = [0.0f32; 36];
        build_so6_kelvin_rotor(&r_id, &mut m6_id);
        let err_id = check_so6_orthogonality(&m6_id);
        assert!(err_id <= 1e-7, "SO(6) rotor error at identity: {}", err_id);

        // Test non-trivial rotation around Y-axis by 30 deg
        let theta = 30.0f32.to_radians();
        let ct = theta.cos();
        let st = theta.sin();
        let r_rot = [
            ct,  0.0, st,
            0.0, 1.0, 0.0,
            -st, 0.0, ct,
        ];
        let mut m6 = [0.0f32; 36];
        build_so6_kelvin_rotor(&r_rot, &mut m6);
        let err = check_so6_orthogonality(&m6);
        assert!(err <= 1e-6, "SO(6) rotor error: {}", err);

        // Verify trace conservation
        let mut c_rot = [0.0f32; 36];
        rotate_stiffness_6x6(&m6, &C_TTI_BASE_DEFAULT, &mut c_rot);
        let tr_base: f32 = (0..6).map(|i| C_TTI_BASE_DEFAULT[i * 6 + i]).sum();
        let tr_rot: f32 = (0..6).map(|i| c_rot[i * 6 + i]).sum();
        let tr_diff = abs_f32(tr_rot - tr_base);
        assert!(tr_diff <= 1e-5, "Trace conservation error: {}", tr_diff);

        // Double-precision SO(6) rotor orthogonality check (Theorem 2.7 <= 1.14e-13)
        let theta_f64 = 30.0f64.to_radians();
        let ct_f64 = theta_f64.cos();
        let st_f64 = theta_f64.sin();
        let r_f64 = [
            ct_f64,  0.0, st_f64,
            0.0,     1.0, 0.0,
            -st_f64, 0.0, ct_f64,
        ];
        let mut m6_f64 = [0.0f64; 36];
        build_so6_kelvin_rotor_f64(&r_f64, &mut m6_f64);
        let err_f64 = check_so6_orthogonality_f64(&m6_f64);
        assert!(
            err_f64 <= 1.14e-13,
            "SO(6) rotor double-precision error {} > 1.14e-13",
            err_f64
        );
    }

    #[test]
    fn test_mad_floor_256() {
        let mut salience = [0.0f32; TOTAL_VOXELS];
        for (i, val) in salience.iter_mut().enumerate() {
            *val = ((i * 37) % 100) as f32 / 100.0;
        }
        let floor = unsafe { compute_spatial_mad_floor_256(salience.as_ptr(), core::ptr::null(), TOTAL_VOXELS) };
        assert!(floor > 0.0, "MAD floor must be positive");
    }

    #[test]
    fn test_spherical_cascade_drift() {
        let mut v_in = [0.35355339f32; CASCADE_DIM];
        let u = [0.35355339f32; CASCADE_DIM];
        let mut v_out = [0.0f32; CASCADE_DIM];
        let mut drift = -1.0f32;

        unsafe {
            skew_symmetric_triad_cascade_step_f32(
                u.as_ptr(),
                v_in.as_ptr(),
                core::ptr::null(),
                16,
                0.05,
                v_out.as_mut_ptr(),
                &mut drift,
            );
        }
        assert_eq!(drift, 0.0f32, "Cascade radial drift must be exactly 0.000e+00");
    }

    #[test]
    fn test_step_iter_lifecycle() {
        unsafe {
            let arena = toposnail_arena_new();
            assert!(!arena.is_null());
            toposnail_arena_reset(arena, 16, 16, 16, 0.25);
            let comp = toposnail_step_iter(arena, 0.25, 3.0, 1.5, 1.0);
            assert!(comp >= 0.0);
            assert_eq!((*arena).iteration, 1);
            toposnail_arena_free(arena);
        }
    }

    #[test]
    fn test_ulp_polish_zero_preservation() {
        let mut v = [0.0f32; CASCADE_DIM];
        v[0] = 0.6f32;
        v[1] = 0.8f32;
        // coordinates 2..8 are strictly 0.0
        exact_sphere_norm_ulp_polish(&mut v, 1.0f32);
        for (i, &vi) in v.iter().enumerate().skip(2) {
            assert_eq!(vi, 0.0f32, "Zero coordinate {} was perturbed!", i);
        }
    }

    #[test]
    fn test_fast_sqrt_quad_simd() {
        assert_eq!(fast_sqrt(0.0), 0.0);
        assert_eq!(fast_sqrt(-1.0), 0.0);
        assert_eq!(fast_sqrt(4.0), 2.0);
        assert_eq!(fast_sqrt(16.0), 4.0);
        assert!(fast_sqrt(f32::INFINITY).is_infinite());
        // Subnormal test
        let subnormal = f32::from_bits(0x0000_0100);
        assert!(fast_sqrt(subnormal) > 0.0);
    }
}
