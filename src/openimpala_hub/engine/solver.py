"""End-to-end `OpenImpala` transport, stereology, field post-processing, and BPX pipeline."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import os
import threading
import time
import uuid

import numpy as np
import basecoat
from basecoat import Volume
import openimpala as oi

from .. import __version__ as _HUB_VERSION
from ..contracts import (
    AxisTransportSummary,
    MicrostructureSpec,
    SolveRunSummary,
    SolveSpec,
)
from .bridge import (
    enforce_solve_grid_cap,
    extract_phase_volume,
    resolve_voxel_size_metres,
    solver_input_for_openimpala,
)
from .generators import load_or_generate_microstructure
from .steering import (
    compute_cell_resistance,
    compute_diffusive_flux_field,
    find_flux_bottleneck_rois,
)

SOLVER_LOCK = threading.RLock()


@contextmanager
def _scoped_session(*, use_pure: bool = False):
    with SOLVER_LOCK:
        prev_env = os.environ.get("OPENIMPALA_BACKEND")
        try:
            if use_pure:
                os.environ["OPENIMPALA_BACKEND"] = "pure"
            with oi.Session():
                yield
        finally:
            if use_pure:
                if prev_env is None:
                    os.environ.pop("OPENIMPALA_BACKEND", None)
                else:
                    os.environ["OPENIMPALA_BACKEND"] = prev_env


@dataclass
class ExecutedRun:
    """In-memory handle pairing a `SolveRunSummary` with calibrated 3D phase, potential, and flux fields."""

    summary: SolveRunSummary
    phase_volume: Volume
    potential_field: np.ndarray | None
    flux_field: np.ndarray | None


def _safe_float(val: object | None) -> float | None:
    if val is None:
        return None
    f = float(val)  # type: ignore[arg-type]
    return f if np.isfinite(f) else None


def run_transport_pipeline(
    micro_spec: MicrostructureSpec,
    solve_spec: SolveSpec | None = None,
    *,
    existing_volume: Volume | None = None,
) -> ExecutedRun:
    """Generate/load a microstructure, enforce physical units & grid caps, and execute `OpenImpala`."""
    s_spec = solve_spec or SolveSpec()

    t0 = time.perf_counter()
    if existing_volume is not None:
        raw_vol = extract_phase_volume(existing_volume)
        origin = str(raw_vol.provenance.get("data_origin", "user_upload"))  # type: ignore[assignment]
        if origin not in ("synthetic", "porousdiff", "benchmark_xct", "user_upload"):
            origin = "user_upload"
        precision_tag = f"{raw_vol.dtype.name} supplied Volume"
        orig_shape = raw_vol.shape
    else:
        raw_vol, origin, precision_tag, orig_shape = load_or_generate_microstructure(micro_spec)
    gen_time_s = time.perf_counter() - t0

    # Fail loudly if uncalibrated (`unit='voxel'`) BEFORE touching solver kernels
    resolve_voxel_size_metres(raw_vol)

    capped_vol, coarsened_note = enforce_solve_grid_cap(
        raw_vol,
        max_side=s_spec.max_solve_side,
        policy=micro_spec.coarsen_policy,
    )
    dz_m, dy_m, dx_m = resolve_voxel_size_metres(capped_vol)
    vz_um = capped_vol.spacing_in("um")

    use_pure = s_spec.solver == "scipy"
    solver_name = "mlmg" if s_spec.solver == "scipy" else s_spec.solver

    t_solve_start = time.perf_counter()
    with _scoped_session(use_pure=use_pure):
        binfo = oi.build_info()
        actual_backend = "scipy-pure" if use_pure else str(binfo.get("backend", "unknown"))
        data_arg, vox_kw = solver_input_for_openimpala(capped_vol)

        # Phase volume fractions
        arr = capped_vol.data
        unique_phases = sorted(int(p) for p in np.unique(arr))
        lbl_map = capped_vol.label_names or micro_spec.phase_names
        phase_fractions: dict[str, float] = {}
        for pid in unique_phases:
            vf_res = oi.volume_fraction(data_arg, phase=pid)
            pname = lbl_map.get(pid, f"phase_{pid}")
            phase_fractions[f"{pid}:{pname}"] = round(float(vf_res.fraction), 6)

        vf_cond = oi.volume_fraction(data_arg, phase=s_spec.conducting_phase)
        porosity = float(vf_cond.fraction)

        # Cauchy-Crofton Specific Surface Area (pore <-> solid backbone & pore <-> active coating)
        primary_solid = s_spec.solid_phases[0] if s_spec.solid_phases else 1
        ssa_res = oi.specific_surface_area(
            data_arg,
            phase_a=s_spec.conducting_phase,
            phase_b=primary_solid,
            **vox_kw,
        )
        ssa_total = float(ssa_res.ssa_corrected)

        active_ssa = ssa_total
        if 2 in unique_phases and 2 in s_spec.solid_phases:
            ssa_coat = oi.specific_surface_area(
                data_arg,
                phase_a=s_spec.conducting_phase,
                phase_b=2,
                **vox_kw,
            )
            ssa_total += float(ssa_coat.ssa_corrected)
            # Active electrochemical interface weights catalyst/CBD coating plus exposed carbon surface
            active_ssa = float(ssa_coat.ssa_corrected) + 0.35 * float(ssa_res.ssa_corrected)

        # Slice-by-slice through-thickness porosity profile along field_direction
        prof_res = oi.through_thickness_profile(
            data_arg,
            phase=s_spec.conducting_phase,
            direction=s_spec.field_direction,
            **vox_kw,
        )
        prof_z_um = [round(float(p) * 1e6, 3) for p in prof_res.positions]
        prof_eps = [round(float(v), 5) for v in prof_res.profile]

        # Directional transport solves (single-phase or composite multiphase)
        directional: dict[str, AxisTransportSummary] = {}
        potential_field: np.ndarray | None = None

        for d in s_spec.directions:
            want_field = bool(s_spec.return_fields and d == s_spec.field_direction)
            if s_spec.phase_diffusivities is not None:
                # Composite multiphase steady-state transport
                tres = oi.tortuosity(
                    data_arg,
                    phase_diffusivities=s_spec.phase_diffusivities,
                    direction=d,
                    solver=solver_name,
                    return_field=want_field,
                    **vox_kw,
                )
                perc_flag = bool(tres.effective_diffusivity and tres.effective_diffusivity > 1e-12)
                act_vf = float(tres.active_volume_fraction)
            else:
                perc = oi.percolation_check(
                    data_arg,
                    phase=s_spec.conducting_phase,
                    direction=d,
                )
                perc_flag = bool(perc.percolates)
                act_vf = float(perc.active_volume_fraction)
                if perc_flag:
                    tres = oi.tortuosity(
                        data_arg,
                        phase=s_spec.conducting_phase,
                        direction=d,
                        solver=solver_name,
                        return_field=want_field,
                        **vox_kw,
                    )
                else:
                    tres = None

            if tres is not None:
                if want_field and tres.solution_field is not None:
                    potential_field = np.asarray(tres.solution_field, dtype=np.float32)
                deff_val = (
                    float(tres.effective_diffusivity)
                    if tres.effective_diffusivity is not None
                    else (act_vf / float(tres.tortuosity) if tres.tortuosity > 0 else 0.0)
                )
                directional[d] = AxisTransportSummary(
                    direction=d,
                    percolates=perc_flag,
                    active_volume_fraction=round(act_vf, 6),
                    tortuosity=_safe_float(round(float(tres.tortuosity), 6)),
                    effective_diffusivity=round(deff_val, 6),
                    macmullin_number=_safe_float(
                        round(float(tres.macmullin_number), 6)
                        if np.isfinite(tres.macmullin_number)
                        else None
                    ),
                    bruggeman_exponent=(
                        round(float(tres.bruggeman_exponent), 6)
                        if tres.bruggeman_exponent is not None
                        else None
                    ),
                    solver_converged=bool(tres.solver_converged),
                    iterations=int(tres.iterations),
                    residual_norm=float(tres.residual_norm),
                )
            else:
                directional[d] = AxisTransportSummary(
                    direction=d,
                    percolates=False,
                    active_volume_fraction=0.0,
                    tortuosity=None,
                    effective_diffusivity=0.0,
                    macmullin_number=None,
                    bruggeman_exponent=None,
                    solver_converged=True,
                    iterations=0,
                    residual_norm=0.0,
                )

        # Solid electronic wiring along Z (for coupled R_cell(sigma) digital-twin trade-off)
        solid_percolates = False
        solid_sigma_z = 0.0
        if s_spec.compute_solid_wiring and s_spec.solid_phases:
            solid_binary = np.isin(arr, s_spec.solid_phases).astype(np.int32)
            if 0.0 < float(np.mean(solid_binary)) < 1.0:
                s_perc = oi.percolation_check(solid_binary, phase=1, direction="z")
                solid_percolates = bool(s_perc.percolates)
                if solid_percolates:
                    s_tau = oi.tortuosity(
                        solid_binary,
                        phase=1,
                        direction="z",
                        solver=solver_name,
                        voxel_size=(dz_m, dy_m, dx_m),
                    )
                    solid_sigma_z = (
                        float(s_tau.effective_diffusivity)
                        if s_tau.effective_diffusivity is not None
                        else float(s_tau.active_volume_fraction) / max(float(s_tau.tortuosity), 1.0)
                    )
            elif float(np.mean(solid_binary)) == 1.0:
                solid_percolates = True
                solid_sigma_z = 1.0

        # Optional 3x3 periodic homogenisation tensor
        homog_tensor: list[list[float]] | None = None
        if s_spec.compute_homogenisation:
            if s_spec.phase_diffusivities is not None:
                h_res = oi.effective_diffusivity(
                    data_arg,
                    phase_diffusivities=s_spec.phase_diffusivities,
                    solver="hypre" if solver_name == "hypre" else "auto",
                    **vox_kw,
                )
            else:
                h_res = oi.effective_diffusivity(
                    data_arg,
                    phase=s_spec.conducting_phase,
                    solver="hypre" if solver_name == "hypre" else "auto",
                    **vox_kw,
                )
            homog_tensor = [
                [round(float(h_res.tensor[r, c]), 6) for c in range(3)] for r in range(3)
            ]

        # Build PyBaMM & Faraday BPX dictionaries directly from already-solved metrics (zero redundant PDE solves)
        pybamm_params: dict[str, object] = {}
        bpx_params: dict[str, object] = {}
        primary_dir = s_spec.field_direction
        dir_axes_m = {"z": dz_m, "y": dy_m, "x": dx_m}
        dir_shapes = {"z": capped_vol.shape[0], "y": capped_vol.shape[1], "x": capped_vol.shape[2]}
        p_summary = directional.get(primary_dir)

        pybamm_params["Negative electrode porosity"] = round(porosity, 6)
        pybamm_params["Negative electrode thickness [m]"] = round(
            float(dir_shapes[primary_dir]) * float(dir_axes_m[primary_dir]), 9
        )
        if p_summary and p_summary.bruggeman_exponent is not None:
            pybamm_params["Negative electrode Bruggeman coefficient (electrolyte)"] = (
                p_summary.bruggeman_exponent
            )
        solid_vf = round(1.0 - porosity, 6)
        if solid_vf > 0.0:
            pybamm_params["Negative electrode active material volume fraction"] = solid_vf
            pybamm_params["Negative electrode specific surface area [m-1]"] = round(ssa_total, 2)

        bpx_results = {
            d.upper(): {
                "Deff_ratio": dt.effective_diffusivity,
                "tortuosity": dt.tortuosity,
            }
            for d, dt in directional.items()
        }
        bpx_params = {
            "schema": "openimpala.bpx.v1",
            "sample_id": micro_spec.label,
            "openimpala": {
                "version": oi.__version__,
                "physics_type": "Diffusion",
                "phase_id": int(s_spec.conducting_phase),
                "solver": solver_name,
                "grid": {
                    "nx": int(capped_vol.shape[2]),
                    "ny": int(capped_vol.shape[1]),
                    "nz": int(capped_vol.shape[0]),
                },
                "volume_fraction": round(porosity, 6),
                "results": bpx_results,
                "deff_tensor_xyz": homog_tensor,
            },
            "units": {
                "voxel_size_m": [dz_m, dy_m, dx_m],
                "specific_surface_area": "m^-1",
            },
            "bpx": {
                "Parameter": {
                    "Negative electrode porosity": round(porosity, 6),
                    "Negative electrode active material volume fraction": solid_vf,
                    "Negative electrode surface area to volume ratio [m-1]": round(ssa_total, 2),
                    "Negative electrode diffusivity ratio (through-plane)": (
                        p_summary.effective_diffusivity if p_summary else 0.0
                    ),
                }
            },
        }

    solve_time_s = time.perf_counter() - t_solve_start

    # Compute 3D dimensionless diffusive flux & spatial zoom-ROI recommendations
    flux_field: np.ndarray | None = None
    zoom_rois = []
    if potential_field is not None:
        flux_field = compute_diffusive_flux_field(
            capped_vol,
            potential_field,
            flow_direction=s_spec.field_direction,
            conducting_phase=s_spec.conducting_phase,
            phase_diffusivities=s_spec.phase_diffusivities,
        )
        roi_w = max(8, min(20, min(capped_vol.shape) // 4))
        zoom_rois = find_flux_bottleneck_rois(
            capped_vol,
            flux_field,
            conducting_phase=s_spec.conducting_phase,
            roi_side=roi_w,
            top_k=3,
        )

    # Anisotropy ratio tau_z / mean(tau_x, tau_y)
    aniso_ratio: float | None = None
    tz = directional["z"].tortuosity if "z" in directional else None
    tx = directional["x"].tortuosity if "x" in directional else None
    ty = directional["y"].tortuosity if "y" in directional else None
    if tz is not None and tx is not None and ty is not None and (tx + ty) > 0.0:
        aniso_ratio = round(float(tz) / (0.5 * (float(tx) + float(ty))), 4)

    d_eff_z = directional["z"].effective_diffusivity if "z" in directional else porosity
    thickness_ratio = (
        max(0.35, 1.0 - float(micro_spec.compression_strain))
        if micro_spec.source_kind in ("carbon_felt", "porousdiff_gallery")
        else 1.0
    )
    r_cell = compute_cell_resistance(
        d_eff_z=d_eff_z,
        sigma_eff_z=solid_sigma_z if solid_sigma_z > 0.0 else 0.015,
        active_area_m_inv=active_ssa if active_ssa > 0.0 else 1.0e5,
        thickness_ratio=thickness_ratio,
    )

    summary = SolveRunSummary(
        run_id=uuid.uuid4().hex[:10],
        label=micro_spec.label,
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        data_origin=origin,
        precision_tag=precision_tag,
        source_kind=micro_spec.source_kind,
        shape_zyx=capped_vol.shape,
        original_shape_zyx=orig_shape,
        coarsened_note=coarsened_note,
        voxel_size_um_zyx=(round(vz_um[0], 4), round(vz_um[1], 4), round(vz_um[2], 4)),
        phase_fractions=phase_fractions,
        conducting_phase=s_spec.conducting_phase,
        porosity=round(porosity, 6),
        specific_surface_area_m_inv=round(ssa_total, 2),
        interfacial_area_active_m_inv=round(active_ssa, 2),
        directional=directional,
        anisotropy_ratio_z_over_xy=aniso_ratio,
        solid_through_plane_percolates=solid_percolates,
        solid_effective_conductivity_z=round(solid_sigma_z, 6),
        cell_resistance_fom=round(r_cell, 5),
        homogenised_tensor_xyz=homog_tensor,
        through_thickness_z_um=prof_z_um,
        through_thickness_porosity=prof_eps,
        zoom_rois=zoom_rois,
        pybamm_parameters=dict(pybamm_params),
        bpx_parameters=dict(bpx_params),
        solver_backend=actual_backend,
        openimpala_version=oi.__version__,
        basecoat_version=basecoat.__version__,
        hub_version=_HUB_VERSION,
        generation_time_s=round(gen_time_s, 4),
        solve_time_s=round(solve_time_s, 4),
        microstructure_spec=micro_spec.to_dict(),
        solve_spec=s_spec.to_dict(),
    )

    return ExecutedRun(
        summary=summary,
        phase_volume=capped_vol,
        potential_field=potential_field,
        flux_field=flux_field,
    )
