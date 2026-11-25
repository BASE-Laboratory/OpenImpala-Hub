"""Typed run, result, steering, and provenance contracts for The OpenImpala Hub.

All contracts follow the BASE Lab ecosystem invariants (`ECOSYSTEM.md`):
- Spatial arrays and coordinates follow `(z, y, x)` order with index-0 cell centres.
- Physical units are explicit (`'um'` on `Volume`, converted to SI `'m'` at the solver boundary).
- Every result records explicit `data_origin` (`'synthetic'`, `'porousdiff'`, `'benchmark_xct'`,
  or `'user_upload'`), software versions, solver backend, and runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence

DataOrigin = Literal["synthetic", "porousdiff", "benchmark_xct", "user_upload"]
SourceKind = Literal[
    "carbon_felt",
    "gaussian_blobs",
    "overlapping_spheres",
    "multiphase_electrode",
    "open_cube",
    "benchmark_dataset",
    "porousdiff_gallery",
    "user_upload",
]
DirectionName = Literal["x", "y", "z"]
ElectrodeRole = Literal["negative", "positive", "separator"]


def _normalize_3f(val: float | Sequence[float]) -> tuple[float, float, float]:
    if isinstance(val, (int, float)):
        v = float(val)
        return (v, v, v)
    seq = tuple(float(x) for x in val)
    if len(seq) != 3:
        raise ValueError(f"Expected scalar or 3-tuple, got {val!r}.")
    return (seq[0], seq[1], seq[2])


def _normalize_3i(val: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(val, int):
        v = int(val)
        return (v, v, v)
    seq = tuple(int(x) for x in val)
    if len(seq) != 3:
        raise ValueError(f"Expected int or 3-tuple of ints, got {val!r}.")
    return (seq[0], seq[1], seq[2])


@dataclass(frozen=True)
class MicrostructureSpec:
    """Specification for generating or loading a 3D `(z, y, x)` phase microstructure."""

    source_kind: SourceKind = "carbon_felt"
    label: str = "Anisotropic Carbon Felt (150 kPa)"
    shape: tuple[int, int, int] = (64, 64, 64)
    voxel_size_um: tuple[float, float, float] | None = (1.6, 1.6, 1.6)
    seed: int = 42
    porosity: float = 0.74
    compression_strain: float = 0.18
    clamping_kpa: float = 150.0
    anisotropy_zyx: tuple[float, float, float] = (0.45, 1.8, 1.8)
    fiber_radius_vox: float = 1.9
    coating_fraction: float = 0.08
    blobiness: float = 1.2
    sphere_radius_vox: float = 6.0
    active_fraction: float = 0.52
    binder_fraction: float = 0.12
    dataset_name: str | None = None
    upload_path: str | None = None
    threshold: float | str | None = None
    dark_is_pore: bool = True
    override_calibrated_spacing: bool = False
    coarsen_policy: Literal["crop_center", "mode_downsample", "error"] = "crop_center"
    roi_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None = None
    electrode_role: ElectrodeRole = "negative"
    phase_names: dict[int, str] = field(
        default_factory=lambda: {0: "pore", 1: "solid_backbone", 2: "active_coating"}
    )

    def __post_init__(self) -> None:
        shp = _normalize_3i(self.shape)
        if any(s < 4 for s in shp):
            raise ValueError(f"Microstructure shape dimensions must be >= 4, got {shp}.")
        object.__setattr__(self, "shape", shp)
        if self.voxel_size_um is not None:
            vz = _normalize_3f(self.voxel_size_um)
            if any(v <= 0.0 for v in vz):
                raise ValueError(f"voxel_size_um must be strictly positive, got {vz}.")
            object.__setattr__(self, "voxel_size_um", vz)
        object.__setattr__(self, "anisotropy_zyx", _normalize_3f(self.anisotropy_zyx))
        role = str(self.electrode_role).strip().lower()
        if role not in ("negative", "positive", "separator"):
            raise ValueError(
                f"Unsupported electrode_role {self.electrode_role!r}; "
                "expected 'negative', 'positive', or 'separator'."
            )
        object.__setattr__(self, "electrode_role", role)
        object.__setattr__(
            self,
            "phase_names",
            {int(k): str(v) for k, v in dict(self.phase_names).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["phase_names"] = {str(k): v for k, v in self.phase_names.items()}
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MicrostructureSpec:
        payload = dict(data)
        if "shape" in payload:
            payload["shape"] = tuple(int(x) for x in payload["shape"])
        if payload.get("voxel_size_um") is not None:
            payload["voxel_size_um"] = _normalize_3f(payload["voxel_size_um"])
        if "anisotropy_zyx" in payload:
            payload["anisotropy_zyx"] = _normalize_3f(payload["anisotropy_zyx"])
        if payload.get("roi_zyx") is not None:
            payload["roi_zyx"] = tuple((int(a), int(b)) for a, b in payload["roi_zyx"])
        if "phase_names" in payload and isinstance(payload["phase_names"], Mapping):
            payload["phase_names"] = {int(k): str(v) for k, v in payload["phase_names"].items()}
        return cls(**payload)


@dataclass(frozen=True)
class SolveSpec:
    """Configuration for an `OpenImpala` transport and stereology evaluation."""

    conducting_phase: int = 0
    solid_phases: tuple[int, ...] = (1, 2)
    phase_diffusivities: dict[int, float] | None = None
    directions: tuple[DirectionName, ...] = ("x", "y", "z")
    solver: Literal["mlmg", "hypre", "scipy"] = "mlmg"
    compute_homogenisation: bool = False
    compute_solid_wiring: bool = True
    compute_rev_study: bool = False
    return_fields: bool = True
    field_direction: DirectionName = "z"
    max_solve_side: int | None = None
    r_cell_weights: tuple[float, float, float] = (0.58, 0.27, 0.15)
    sigma_over_kappa_bulk: float = 16.0

    def __post_init__(self) -> None:
        dirs = tuple(str(d).lower() for d in self.directions)
        for d in dirs:
            if d not in ("x", "y", "z"):
                raise ValueError(f"Unsupported direction {d!r}; expected 'x', 'y', or 'z'.")
        object.__setattr__(self, "directions", dirs)
        object.__setattr__(self, "solid_phases", tuple(int(p) for p in self.solid_phases))
        object.__setattr__(self, "r_cell_weights", _normalize_3f(self.r_cell_weights))
        object.__setattr__(self, "sigma_over_kappa_bulk", float(self.sigma_over_kappa_bulk))
        if self.phase_diffusivities is not None:
            object.__setattr__(
                self,
                "phase_diffusivities",
                {int(k): float(v) for k, v in self.phase_diffusivities.items()},
            )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["r_cell_weights"] = list(self.r_cell_weights)
        if self.phase_diffusivities is not None:
            out["phase_diffusivities"] = {
                str(k): float(v) for k, v in self.phase_diffusivities.items()
            }
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SolveSpec:
        payload = dict(data)
        if "directions" in payload:
            payload["directions"] = tuple(payload["directions"])
        if "solid_phases" in payload:
            payload["solid_phases"] = tuple(int(p) for p in payload["solid_phases"])
        if "r_cell_weights" in payload and payload["r_cell_weights"] is not None:
            payload["r_cell_weights"] = _normalize_3f(payload["r_cell_weights"])
        if payload.get("phase_diffusivities") is not None:
            payload["phase_diffusivities"] = {
                int(k): float(v) for k, v in payload["phase_diffusivities"].items()
            }
        return cls(**payload)


@dataclass(frozen=True)
class AxisTransportSummary:
    """Directional transport summary for a single Cartesian axis."""

    direction: DirectionName
    percolates: bool
    active_volume_fraction: float
    tortuosity: float | None
    effective_diffusivity: float
    macmullin_number: float | None
    bruggeman_exponent: float | None
    solver_converged: bool
    iterations: int
    residual_norm: float


@dataclass(frozen=True)
class ZoomROI:
    """Spatial 3D sub-volume recommended for high-resolution operando zoom XCT / XRD-CT."""

    rank: int
    bounds_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    center_um_zyx: tuple[float, float, float]
    extent_um_zyx: tuple[float, float, float]
    mean_flux_crowding: float
    peak_flux_magnitude: float
    local_porosity: float
    constriction_index: float
    rationale: str


@dataclass(frozen=True)
class BayesianSetpointSuggestion:
    """1D Gaussian Process recommendation for the next physical compression / porosity scan."""

    parameter_name: str
    parameter_unit: str
    suggested_value: float
    predicted_r_cell_mean: float
    predicted_r_cell_std: float
    expected_improvement: float
    current_best_parameter: float
    current_best_r_cell: float
    grid_x: list[float]
    gp_mean: list[float]
    gp_lower_95: list[float]
    gp_upper_95: list[float]
    ei_curve: list[float]
    observed_x: list[float]
    observed_r_cell: list[float]
    rationale: str


@dataclass(frozen=True)
class SolveRunSummary:
    """Serializable summary of an executed microstructure + OpenImpala evaluation."""

    run_id: str
    label: str
    timestamp_utc: str
    data_origin: DataOrigin
    precision_tag: str
    source_kind: SourceKind
    shape_zyx: tuple[int, int, int]
    original_shape_zyx: tuple[int, int, int]
    coarsened_note: str | None
    voxel_size_um_zyx: tuple[float, float, float]
    phase_fractions: dict[str, float]
    conducting_phase: int
    porosity: float
    specific_surface_area_m_inv: float
    interfacial_area_active_m_inv: float
    directional: dict[str, AxisTransportSummary]
    anisotropy_ratio_z_over_xy: float | None
    solid_through_plane_percolates: bool
    solid_effective_conductivity_z: float
    cell_resistance_fom: float
    homogenised_tensor_xyz: list[list[float]] | None
    through_thickness_z_um: list[float]
    through_thickness_porosity: list[float]
    zoom_rois: list[ZoomROI]
    pybamm_parameters: dict[str, Any]
    bpx_parameters: dict[str, Any]
    solver_backend: str
    openimpala_version: str
    basecoat_version: str
    hub_version: str
    generation_time_s: float
    solve_time_s: float
    microstructure_spec: dict[str, Any]
    solve_spec: dict[str, Any]
    electrode_role: ElectrodeRole = "negative"
    rev_sizes_vox: list[int] | None = None
    rev_deff_mean: list[float] | None = None
    rev_deff_std: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
