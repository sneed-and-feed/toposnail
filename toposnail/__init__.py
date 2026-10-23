"""
toposnail: Beltrami-Triad 3D Anisotropic Topology, Michell Truss & #![no_std] Rust SIMD Optimizer + 3D WebGL .STL Studio

Powered by machine-checked OcclusionTriad.lean invariants (Theorems 2.1-2.8).

Key Modules:
- elasticity: Matrix-Free 3D Hexahedral EBE Elasticity, SO(6) Kelvin-Mandel Stiffness Rotors,
  Thomsen TTI Elastic Anisotropy, and Tsai-Hill Yield Envelopes.
- optimizer: TopoSnail 3D Anisotropic Continuum Topology Optimizer with 4-stage Beltrami-Morse
  thin-strut shielding, 1D Marginal W_1 seeding, and Jordan cavity enforcement.
"""

from __future__ import annotations

from .blind_cad_benchmark import (
    audit_blind_cad_part_intrinsic,
    run_blind_cad_corpus_benchmark,
    run_blind_cad_part_benchmark,
)
from .blind_cad_corpus import (
    generate_blind_cad_corpus,
    load_blind_cad_mesh,
    recognize_blind_cad_features,
)
from .elasticity import (
    EBERelaxationResult,
    ElasticMaterialProperties,
    HexahedralMesh3D,
    ThomsenTTIAnisotropicElasticEngine,
)
from .optimizer import (
    BeltramiMorseStrutFilter,
    BoundaryConditions3D,
    JordanCavityEnforcer,
    OptimizerStepResult,
    TemporalLoadContinuitySplitter,
    TopoSnailOptimizer,
)

run_blind_cad_benchmark_suite = run_blind_cad_corpus_benchmark

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "HexahedralMesh3D",
    "ElasticMaterialProperties",
    "EBERelaxationResult",
    "ThomsenTTIAnisotropicElasticEngine",
    "BoundaryConditions3D",
    "OptimizerStepResult",
    "TemporalLoadContinuitySplitter",
    "BeltramiMorseStrutFilter",
    "JordanCavityEnforcer",
    "TopoSnailOptimizer",
    "generate_blind_cad_corpus",
    "load_blind_cad_mesh",
    "recognize_blind_cad_features",
    "run_blind_cad_benchmark_suite",
    "run_blind_cad_corpus_benchmark",
    "run_blind_cad_part_benchmark",
    "audit_blind_cad_part_intrinsic",
]
