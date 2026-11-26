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
from .electrochem import evaluate_cell_electrochemistry
from .generators import load_or_generate_microstructure
from .mechanics import run_optional_mechanics
from .morphology import compute_morphology_report
from .porousdiff_bridge import kinematic_compress_volume
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
    """In-memory handle pairing a `SolveRunSummary` with calibrated 3D phase, potential, flux, and stress fields."""

    summary: SolveRunSummary
    phase_volume: Volume
    potential_field: np.ndarray | None
    flux_field: np.ndarray | None
    vm_stress_field: np.ndarray | None = None


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
    """Generate/load a microstructure, enforce physical units & grid caps, and execute `OpenImpala` (+ `SMS`)."""
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

    if (
        micro_spec.virtual_compression_strain > 1e-4
        and micro_spec.source_kind != "carbon_felt"
    ):
        raw_vol = kinematic_compress_volume(
            raw_vol,
            micro_spec.virtual_compression_strain,
            conducting_phase=s_spec.conducting_phase,
        )
        precision_tag = (
            f"{precision_tag} + virtual Z-compression (eps_z={micro_spec.virtual_compression_strain:.2f})"
        )
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

        # Cauchy-Crofton Specific Surface Area (total and percolation-filtered active SSA)
        counter_phases = [p for p in unique_phases if p != s_spec.conducting_phase]
        if counter_phases:
            primary_other = next(
                (p for p in s_spec.solid_phases if p in counter_phases),
                counter_phases[0],
            )
            ssa_res = oi.specific_surface_area(
                data_arg,
                phase_a=s_spec.conducting_phase,
                phase_b=primary_other,
                **vox_kw,
            )
            ssa_act_res = oi.specific_surface_area(
                data_arg,
                phase_a=s_spec.conducting_phase,
                phase_b=primary_other,
                active_only=True,
                direction=s_spec.field_direction,
                active_phase=s_spec.conducting_phase,
                **vox_kw,
            )
            ssa_total = float(ssa_res.ssa_corrected)
            active_ssa = float(ssa_act_res.ssa_corrected)
            if (
                2 in counter_phases
                and primary_other != 2
                and 2 in s_spec.solid_phases
            ):
                ssa_coat = oi.specific_surface_area(
                    data_arg,
                    phase_a=s_spec.conducting_phase,
                    phase_b=2,
                    **vox_kw,
                )
                ssa_coat_act = oi.specific_surface_area(
                    data_arg,
                    phase_a=s_spec.conducting_phase,
                    phase_b=2,
                    active_only=True,
                    direction=s_spec.field_direction,
                    active_phase=s_spec.conducting_phase,
                    **vox_kw,
                )
                ssa_total += float(ssa_coat.ssa_corrected)
                # Active electrochemical interface weights percolating catalyst/CBD coating plus exposed carbon surface
                active_ssa = float(ssa_coat_act.ssa_corrected) + 0.35 * float(ssa_act_res.ssa_corrected)
        else:
            ssa_total = 0.0
            active_ssa = 0.0

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
        native_flux_mag_si: np.ndarray | None = None

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
                    return_flux=want_field,
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
                        return_flux=want_field,
                        **vox_kw,
                    )
                else:
                    tres = None

            if tres is not None:
                if want_field and tres.solution_field is not None:
                    potential_field = np.asarray(tres.solution_field, dtype=np.float32)
                if want_field and tres.flux_magnitude is not None:
                    native_flux_mag_si = np.asarray(tres.flux_magnitude, dtype=np.float64)
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

        # Optional subvolume Representative Elementary Volume (REV) convergence study
        rev_sizes_vox: list[int] | None = None
        rev_deff_mean: list[float] | None = None
        rev_deff_std: list[float] | None = None
        if s_spec.compute_rev_study:
            min_dim = int(min(capped_vol.shape))
            if min_dim >= 12:
                raw_sizes = sorted(
                    {
                        max(8, min_dim // 3),
                        max(12, (2 * min_dim) // 3),
                        min_dim,
                    }
                )
                rev_res = oi.rev_study(
                    data_arg,
                    sizes=raw_sizes,
                    phase=s_spec.conducting_phase,
                    num_samples=3,
                    solver="hypre" if solver_name == "hypre" else "auto",
                    **vox_kw,
                )
                ax_diag_idx = {"x": 0, "y": 1, "z": 2}[s_spec.field_direction]
                rev_sizes_vox = [int(sz) for sz in rev_res.sizes]
                rev_deff_mean = [
                    round(float(rev_res.mean_diagonal[sz][ax_diag_idx]), 6)
                    for sz in rev_res.sizes
                ]
                rev_deff_std = [
                    round(float(rev_res.std_diagonal[sz][ax_diag_idx]), 6)
                    for sz in rev_res.sizes
                ]

        # Build PyBaMM & Faraday BPX dictionaries dynamically from electrode_role
        pybamm_params: dict[str, object] = {}
        bpx_params: dict[str, object] = {}
        primary_dir = s_spec.field_direction
        dir_axes_m = {"z": dz_m, "y": dy_m, "x": dx_m}
        dir_shapes = {"z": capped_vol.shape[0], "y": capped_vol.shape[1], "x": capped_vol.shape[2]}
        p_summary = directional.get(primary_dir)

        role_map = {
            "negative": "Negative electrode",
            "positive": "Positive electrode",
            "separator": "Separator",
        }
        role_prefix = role_map.get(micro_spec.electrode_role, "Negative electrode")
        is_separator = micro_spec.electrode_role == "separator"

        pybamm_params[f"{role_prefix} porosity"] = round(porosity, 6)
        pybamm_params[f"{role_prefix} thickness [m]"] = round(
            float(dir_shapes[primary_dir]) * float(dir_axes_m[primary_dir]), 9
        )
        if p_summary and p_summary.bruggeman_exponent is not None:
            pybamm_params[f"{role_prefix} Bruggeman coefficient (electrolyte)"] = (
                p_summary.bruggeman_exponent
            )
        solid_vf = round(1.0 - porosity, 6)
        if solid_vf > 0.0 and not is_separator:
            pybamm_params[f"{role_prefix} active material volume fraction"] = solid_vf
            pybamm_params[f"{role_prefix} specific surface area [m-1]"] = round(ssa_total, 2)

        bpx_results = {
            d.upper(): {
                "Deff_ratio": dt.effective_diffusivity,
                "tortuosity": dt.tortuosity,
            }
            for d, dt in directional.items()
        }
        bpx_param_block: dict[str, object] = {
            f"{role_prefix} porosity": round(porosity, 6),
            f"{role_prefix} diffusivity ratio (through-plane)": (
                p_summary.effective_diffusivity if p_summary else 0.0
            ),
        }
        if not is_separator:
            bpx_param_block[f"{role_prefix} active material volume fraction"] = solid_vf
            bpx_param_block[f"{role_prefix} surface area to volume ratio [m-1]"] = round(
                ssa_total, 2
            )

        bpx_params = {
            "schema": "openimpala.bpx.v1",
            "sample_id": micro_spec.label,
            "electrode_role": micro_spec.electrode_role,
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
            "bpx": {"Parameter": bpx_param_block},
        }

        morph_dict: dict[str, object] | None = None
        if s_spec.compute_morphology:
            morph_rep = compute_morphology_report(
                capped_vol,
                conducting_phase=s_spec.conducting_phase,
                flow_direction=s_spec.field_direction,
                use_openimpala_psd=True,
            )
            morph_dict = morph_rep.to_dict()

    mech_dict: dict[str, object] | None = None
    vm_stress_field: np.ndarray | None = None
    if s_spec.compute_mechanics:
        with SOLVER_LOCK:
            mech_rep, vm_stress_field = run_optional_mechanics(
                capped_vol,
                solid_phases=s_spec.solid_phases,
                direction=s_spec.field_direction,
                bulk_youngs_modulus_gpa=s_spec.bulk_youngs_modulus_gpa,
            )
            if mech_rep is not None:
                mech_dict = mech_rep.to_dict()

    solve_time_s = time.perf_counter() - t_solve_start

    flow_axis_idx = {"z": 0, "y": 1, "x": 2}[s_spec.field_direction]
    flow_sp_m = (dz_m, dy_m, dx_m)[flow_axis_idx]
    thickness_m = float(capped_vol.shape[flow_axis_idx]) * float(flow_sp_m)

    # Compute 3D dimensionless diffusive flux (preferring native OpenImpala v5.2 flux_magnitude)
    # & spatial zoom-ROI recommendations
    flux_field: np.ndarray | None = None
    zoom_rois = []
    if native_flux_mag_si is not None:
        flux_field = (native_flux_mag_si * thickness_m).astype(np.float32)
    elif potential_field is not None:
        flux_field = compute_diffusive_flux_field(
            capped_vol,
            potential_field,
            flow_direction=s_spec.field_direction,
            conducting_phase=s_spec.conducting_phase,
            phase_diffusivities=s_spec.phase_diffusivities,
        )
    if flux_field is not None:
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
    w_ion, w_elec, w_kin = s_spec.r_cell_weights
    r_cell = compute_cell_resistance(
        d_eff_z=d_eff_z,
        sigma_eff_z=solid_sigma_z if solid_sigma_z > 0.0 else 0.015,
        active_area_m_inv=active_ssa if active_ssa > 0.0 else 1.0e5,
        thickness_ratio=thickness_ratio,
        sigma_over_kappa_bulk=s_spec.sigma_over_kappa_bulk,
        w_ion=w_ion,
        w_elec=w_elec,
        w_kin=w_kin,
    )
    echem_rep = evaluate_cell_electrochemistry(
        porosity=porosity,
        d_eff_z=d_eff_z,
        specific_surface_area_m_inv=active_ssa if active_ssa > 0.0 else 8.0e4,
        solid_sigma_eff_z=solid_sigma_z if solid_sigma_z > 0.0 else 0.05,
        thickness_m=thickness_m,
        chemistry_key=s_spec.chemistry_preset,
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
        electrode_role=micro_spec.electrode_role,
        rev_sizes_vox=rev_sizes_vox,
        rev_deff_mean=rev_deff_mean,
        rev_deff_std=rev_deff_std,
        morphology=morph_dict,
        electrochem=echem_rep.to_dict(),
        mechanics=mech_dict,
    )

    return ExecutedRun(
        summary=summary,
        phase_volume=capped_vol,
        potential_field=potential_field,
        flux_field=flux_field,
        vm_stress_field=vm_stress_field,
    )
