"""Curated 1-click research and teaching presets for The OpenImpala Hub."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import MicrostructureSpec, SolveSpec
from .engine.generators import strain_from_clamping_kpa


@dataclass(frozen=True)
class HubPreset:
    """Named preset pairing a `MicrostructureSpec` and `SolveSpec` with reviewer/student context."""

    key: str
    title: str
    badge: str
    summary: str
    microstructure: MicrostructureSpec
    solve: SolveSpec


def make_carbon_felt_spec(
    clamping_kpa: float = 150.0,
    *,
    shape: tuple[int, int, int] = (64, 64, 64),
    seed: int = 42,
) -> MicrostructureSpec:
    """Build a calibrated `MicrostructureSpec` for fibrous carbon felt at `clamping_kpa`."""
    strain = strain_from_clamping_kpa(clamping_kpa)
    return MicrostructureSpec(
        source_kind="carbon_felt",
        label=f"Anisotropic Carbon Felt ({clamping_kpa:.0f} kPa)",
        shape=shape,
        voxel_size_um=(1.6, 1.6, 1.6),
        seed=seed,
        porosity=0.84,
        compression_strain=round(strain, 4),
        clamping_kpa=float(clamping_kpa),
        anisotropy_zyx=(0.42, 1.8, 1.8),
        fiber_radius_vox=1.9,
        coating_fraction=0.07,
        phase_names={0: "pore", 1: "carbon_fiber", 2: "active_coating"},
    )


PRESETS: dict[str, HubPreset] = {
    "carbon_felt_compression": HubPreset(
        key="carbon_felt_compression",
        title="1. Carbon Felt Virtual Compression",
        badge="Research · Digital Twin",
        summary=(
            "Anisotropic non-woven fibrous electrode with surface-supported catalyst coating "
            "under mechanical Z-clamping (150 kPa). Demonstrates XY vs through-plane (Z) "
            "tortuosity anisotropy, throat current crowding, and Bayesian clamping optimisation."
        ),
        microstructure=make_carbon_felt_spec(150.0, shape=(64, 64, 64), seed=42),
        solve=SolveSpec(
            conducting_phase=0,
            solid_phases=(1, 2),
            directions=("x", "y", "z"),
            solver="mlmg",
            compute_homogenisation=False,
            compute_solid_wiring=True,
            return_fields=True,
            field_direction="z",
        ),
    ),
    "composite_cathode_3phase": HubPreset(
        key="composite_cathode_3phase",
        title="2. 3-Phase Composite Battery Cathode",
        badge="Multiphase · PyBaMM / BPX",
        summary=(
            "Three-phase lithium-ion cathode (0: electrolyte pore D=1.0, 1: insulating active "
            "material D=0.0, 2: nanoporous carbon-binder domain D=0.18) solved with composite "
            "harmonic-mean conductances, 3x3 homogenisation, and Faraday BPX export."
        ),
        microstructure=MicrostructureSpec(
            source_kind="multiphase_electrode",
            label="3-Phase NMC + CBD Cathode",
            shape=(56, 56, 56),
            voxel_size_um=(0.8, 0.8, 0.8),
            seed=108,
            porosity=0.36,
            active_fraction=0.50,
            binder_fraction=0.14,
            sphere_radius_vox=5.5,
            phase_names={
                0: "pore_electrolyte",
                1: "active_material",
                2: "carbon_binder_domain",
            },
        ),
        solve=SolveSpec(
            conducting_phase=0,
            solid_phases=(1, 2),
            phase_diffusivities={0: 1.0, 1: 0.0, 2: 0.18},
            directions=("x", "y", "z"),
            solver="mlmg",
            compute_homogenisation=True,
            compute_solid_wiring=True,
            return_fields=True,
            field_direction="z",
        ),
    ),
    "teaching_analytical_cube": HubPreset(
        key="teaching_analytical_cube",
        title="3a. Teaching: Analytical Open Cube (tau = 1.000)",
        badge="Teaching · Verification",
        summary=(
            "Exact closed-form sanity check for image-based modelling classes: an unobstructed "
            "single-phase domain must yield porosity = 1.000, tortuosity = 1.000 along X/Y/Z, "
            "linear potential phi(z), and uniform dimensionless flux |J/J0| = 1.000 everywhere."
        ),
        microstructure=MicrostructureSpec(
            source_kind="open_cube",
            label="Analytical Open Cube (Verification)",
            shape=(48, 48, 48),
            voxel_size_um=(1.0, 1.0, 1.0),
            seed=1,
            porosity=1.0,
            phase_names={0: "pore"},
        ),
        solve=SolveSpec(
            conducting_phase=0,
            solid_phases=(),
            directions=("x", "y", "z"),
            solver="mlmg",
            compute_homogenisation=False,
            compute_solid_wiring=False,
            return_fields=True,
            field_direction="z",
        ),
    ),
    "teaching_sphere_pack": HubPreset(
        key="teaching_sphere_pack",
        title="3b. Teaching: Benchmark 8-Sphere Array",
        badge="Teaching · Benchmark",
        summary=(
            "Canonical 100^3 SHA-256 verified 8-sphere packing (`openimpala.datasets.sphere_pack`, "
            "porosity 0.888) illustrating streamline bending around insulating spheres and "
            "departure from the idealised Bruggeman relation."
        ),
        microstructure=MicrostructureSpec(
            source_kind="benchmark_dataset",
            label="Benchmark 8-Sphere Array (100^3)",
            dataset_name="sphere_pack",
            shape=(100, 100, 100),
            voxel_size_um=(1.0, 1.0, 1.0),
            seed=7,
            phase_names={0: "pore", 1: "solid_sphere"},
        ),
        solve=SolveSpec(
            conducting_phase=0,
            solid_phases=(1,),
            directions=("x", "y", "z"),
            solver="mlmg",
            compute_homogenisation=False,
            compute_solid_wiring=False,
            return_fields=True,
            field_direction="z",
        ),
    ),
}

DEFAULT_PRESET_KEY = "carbon_felt_compression"
DEFAULT_SWEEP_LOADS_KPA: tuple[float, ...] = (35.0, 150.0, 410.0)
