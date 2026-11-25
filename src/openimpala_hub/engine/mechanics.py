"""Optional Layer-3 3D voxel elasticity bridge to `structural_mechanics_solver` (`SMS`, `Pillar D`).

Orchestrates `sms.effective_modulus` (`BASE-Laboratory/structural_mechanics_solver`) on a
calibrated `basecoat.Volume`:
- Enforces `SMS` isotropic SI-metre spacing & zero-copy `(z, y, x) <-> (x, y, z)` transpose,
- Coarsens grids `> max_mechanics_side` (`28³` default) via majority-mode downsampling so
  single-grid Fortran SOR solves complete in `< 2 s` during interactive web sessions,
- Recovers cell-centred von Mises stress `sigma_vM(z, y, x)` [MPa] and normalised stress
  concentration factor `sigma_vM / |sigma_macro|` for side-by-side slice co-visualisation
  alongside `OpenImpala` diffusive flux crowding `|J / J0|`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import warnings

import numpy as np
from scipy import ndimage
from basecoat import Volume

from ..contracts import DirectionName
from .bridge import enforce_solve_grid_cap


@dataclass(frozen=True)
class MechanicsSummary:
    """Serializable summary of a 3D `SMS` linear elasticity homogenisation & stress solve."""

    backend: str
    sms_version: str
    direction: DirectionName
    solve_grid_zyx: tuple[int, int, int]
    bulk_youngs_modulus_gpa: float
    poisson_ratio: float
    effective_modulus_gpa: float
    relative_stiffness: float
    mean_von_mises_mpa: float
    peak_von_mises_mpa: float
    peak_stress_concentration: float
    converged: bool
    iterations: int
    relative_residual: float
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def has_sms_backend() -> bool:
    """Return True if compiled `sms` (`structural_mechanics_solver`) is importable."""
    try:
        import sms  # type: ignore[import-not-found]

        return hasattr(sms, "effective_modulus")
    except Exception:
        return False


def run_optional_mechanics(
    phase_vol: Volume,
    *,
    solid_phases: tuple[int, ...] = (1, 2),
    direction: DirectionName = "z",
    bulk_youngs_modulus_gpa: float = 12.0,
    poisson_ratio: float = 0.28,
    applied_strain: float = -1.0e-3,
    max_mechanics_side: int = 24,
    rtol: float = 8.0e-4,
    max_iter: int = 450,
) -> tuple[MechanicsSummary | None, np.ndarray | None]:
    """Execute `sms.effective_modulus` if installed, returning `(MechanicsSummary, vm_mpa_zyx)`."""
    if not has_sms_backend():
        return (
            MechanicsSummary(
                backend="sms-not-installed",
                sms_version="none",
                direction=direction,
                solve_grid_zyx=phase_vol.shape,
                bulk_youngs_modulus_gpa=round(float(bulk_youngs_modulus_gpa), 4),
                poisson_ratio=round(float(poisson_ratio), 4),
                effective_modulus_gpa=0.0,
                relative_stiffness=0.0,
                mean_von_mises_mpa=0.0,
                peak_von_mises_mpa=0.0,
                peak_stress_concentration=0.0,
                converged=False,
                iterations=0,
                relative_residual=0.0,
                note=(
                    "Optional .[sms] extra not installed (BASE-Laboratory/structural_mechanics_solver "
                    "is currently private on GitHub)."
                ),
            ),
            None,
        )

    import sms  # type: ignore[import-not-found]

    sms_ver = str(getattr(sms, "__version__", "0.3"))
    dz_m, dy_m, dx_m = phase_vol.spacing_in("m")
    mean_sp = (dz_m + dy_m + dx_m) / 3.0
    if max(abs(dz_m - mean_sp), abs(dy_m - mean_sp), abs(dx_m - mean_sp)) > 0.02 * mean_sp:
        return (
            MechanicsSummary(
                backend="sms-skipped",
                sms_version=sms_ver,
                direction=direction,
                solve_grid_zyx=phase_vol.shape,
                bulk_youngs_modulus_gpa=round(float(bulk_youngs_modulus_gpa), 4),
                poisson_ratio=round(float(poisson_ratio), 4),
                effective_modulus_gpa=0.0,
                relative_stiffness=0.0,
                mean_von_mises_mpa=0.0,
                peak_von_mises_mpa=0.0,
                peak_stress_concentration=0.0,
                converged=False,
                iterations=0,
                relative_residual=0.0,
                note="SMS requires isotropic voxel spacing (dz == dy == dx).",
            ),
            None,
        )

    arr_full = np.asarray(phase_vol.data)
    if not solid_phases:
        return (
            MechanicsSummary(
                backend="sms-analytical-void",
                sms_version=sms_ver,
                direction=direction,
                solve_grid_zyx=phase_vol.shape,
                bulk_youngs_modulus_gpa=round(float(bulk_youngs_modulus_gpa), 4),
                poisson_ratio=round(float(poisson_ratio), 4),
                effective_modulus_gpa=0.0,
                relative_stiffness=0.0,
                mean_von_mises_mpa=0.0,
                peak_von_mises_mpa=0.0,
                peak_stress_concentration=0.0,
                converged=True,
                iterations=0,
                relative_residual=0.0,
                note="Zero solid phases configured (pure fluid/pore domain).",
            ),
            np.zeros(phase_vol.shape, dtype=np.float32),
        )

    solid_mask_full = np.isin(arr_full, solid_phases)
    solid_frac_full = float(np.mean(solid_mask_full))
    if solid_frac_full <= 1e-9:
        return (
            MechanicsSummary(
                backend="sms-analytical-void",
                sms_version=sms_ver,
                direction=direction,
                solve_grid_zyx=phase_vol.shape,
                bulk_youngs_modulus_gpa=round(float(bulk_youngs_modulus_gpa), 4),
                poisson_ratio=round(float(poisson_ratio), 4),
                effective_modulus_gpa=0.0,
                relative_stiffness=0.0,
                mean_von_mises_mpa=0.0,
                peak_von_mises_mpa=0.0,
                peak_stress_concentration=0.0,
                converged=True,
                iterations=0,
                relative_residual=0.0,
                note="Domain contains 0% solid voxels.",
            ),
            np.zeros(phase_vol.shape, dtype=np.float32),
        )

    mech_vol, coarsen_note = enforce_solve_grid_cap(
        phase_vol,
        max_side=max_mechanics_side,
        policy="mode_downsample",
    )
    e0_pa = float(bulk_youngs_modulus_gpa) * 1.0e9

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        e_eff_pa, sres = sms.effective_modulus(
            mech_vol,
            direction=direction,
            youngs_modulus=e0_pa,
            poisson_ratio=float(poisson_ratio),
            applied_strain=float(applied_strain),
            solid_phase=tuple(int(p) for p in solid_phases),
            void_contrast=1.0e-4 if solid_frac_full < 0.999 else 1.0e-6,
            rtol=float(rtol),
            max_iter=int(max_iter),
            check_converged=False,
        )

    # Permute Fortran `(nx, ny, nz)` -> BASE Lab `(nz, ny, nx)` and convert Pa -> MPa
    vm_mpa_coarse = np.asarray(sres.vm_stress.T, dtype=np.float64) * 1.0e-6
    solid_coarse = np.isin(mech_vol.data, solid_phases)
    solid_vm_vals = vm_mpa_coarse[solid_coarse] if np.any(solid_coarse) else vm_mpa_coarse.ravel()

    mean_vm_mpa = float(np.mean(solid_vm_vals)) if solid_vm_vals.size else 0.0
    peak_vm_mpa = float(np.percentile(solid_vm_vals, 99.5)) if solid_vm_vals.size else 0.0
    macro_stress_mpa = max(abs(float(e_eff_pa) * float(applied_strain)) * 1.0e-6, 1.0e-4)
    scf = peak_vm_mpa / macro_stress_mpa if e_eff_pa > 1.0e3 else 0.0

    if mech_vol.shape != phase_vol.shape:
        zoom_zyx = tuple(
            float(phase_vol.shape[i]) / float(mech_vol.shape[i]) for i in range(3)
        )
        vm_mpa_full = ndimage.zoom(vm_mpa_coarse.astype(np.float32), zoom=zoom_zyx, order=1)
    else:
        vm_mpa_full = vm_mpa_coarse.astype(np.float32)

    vm_mpa_full = np.where(solid_mask_full, np.maximum(vm_mpa_full, 0.0), 0.0).astype(np.float32)

    summary = MechanicsSummary(
        backend="sms-fortran",
        sms_version=sms_ver,
        direction=direction,
        solve_grid_zyx=mech_vol.shape,
        bulk_youngs_modulus_gpa=round(float(bulk_youngs_modulus_gpa), 4),
        poisson_ratio=round(float(poisson_ratio), 4),
        effective_modulus_gpa=round(max(0.0, float(e_eff_pa) * 1.0e-9), 5),
        relative_stiffness=round(max(0.0, float(e_eff_pa) / max(e0_pa, 1.0)), 6),
        mean_von_mises_mpa=round(mean_vm_mpa, 4),
        peak_von_mises_mpa=round(peak_vm_mpa, 4),
        peak_stress_concentration=round(scf, 3),
        converged=bool(sres.converged),
        iterations=int(sres.iterations),
        relative_residual=float(sres.relative_residual),
        note=coarsen_note,
    )
    return summary, vm_mpa_full
