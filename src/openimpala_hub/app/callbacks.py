"""Reactive Dash callbacks connecting the UI to the headless OpenImpala-Hub engine (`v0.2`)."""

from __future__ import annotations

import base64
import json
from typing import Any

from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
import numpy as np

from ..contracts import MicrostructureSpec, SolveSpec
from ..engine.bundle import build_report_bundle_bytes
from ..engine.solver import ExecutedRun, run_transport_pipeline
from ..engine.steering import compute_cell_resistance, suggest_next_setpoint_gp
from ..engine.uploads import inspect_staged_volume, stage_upload_bytes
from ..presets import PRESETS, make_carbon_felt_spec
from .figures import (
    SlicePlane,
    build_3d_isosurface_figure,
    build_directional_and_profile_figure,
    build_gp_steering_figure,
    build_slice_figure,
    build_upload_segmentation_preview_figure,
    empty_placeholder_figure,
)
from .layout import history_entry_from_summary, initial_history_records
from .store import RunRegistry


def _history_entry_from_run(run: ExecutedRun) -> dict[str, Any]:
    return history_entry_from_summary(run.summary)


def plane_slider_bounds_for_run(run: ExecutedRun, plane: SlicePlane | str) -> tuple[int, int]:
    """Return `(max_index, default_index)` for `plane` on a `(Nz, Ny, Nx)` volume."""
    nz, ny, nx = run.phase_volume.shape
    pl = str(plane or "xz").lower()
    if pl == "xy":
        axis_len = nz
        roi_idx = 0
    elif pl == "yz":
        axis_len = nx
        roi_idx = 2
    else:
        axis_len = ny
        roi_idx = 1

    max_idx = max(0, int(axis_len) - 1)
    mid_idx = int(axis_len) // 2
    if run.summary.zoom_rois:
        b0, b1 = run.summary.zoom_rois[0].bounds_zyx[roi_idx]
        mid_idx = min(max(0, (int(b0) + int(b1)) // 2), max_idx)
    return max_idx, mid_idx


def _recompute_entry_r_cell(
    entry: dict[str, Any],
    *,
    w_ion: float = 0.58,
    w_elec: float = 0.27,
    w_kin: float = 0.15,
    sigma_over_kappa_bulk: float = 16.0,
) -> float:
    d_eff_z = float(entry.get("d_eff_z", entry.get("porosity", 0.5)))
    sigma_z = float(entry.get("sigma_z", 0.0))
    active_ssa = float(entry.get("active_ssa", 1.0e5))
    thickness_ratio = float(entry.get("thickness_ratio", 1.0))
    return compute_cell_resistance(
        d_eff_z=d_eff_z,
        sigma_eff_z=sigma_z if sigma_z > 0.0 else 0.015,
        active_area_m_inv=active_ssa if active_ssa > 0.0 else 1.0e5,
        thickness_ratio=thickness_ratio,
        sigma_over_kappa_bulk=sigma_over_kappa_bulk,
        w_ion=w_ion,
        w_elec=w_elec,
        w_kin=w_kin,
    )


def compute_gp_for_campaign(
    history: list[dict[str, Any]],
    *,
    active_source_kind: str | None = None,
    w_ion: float = 0.58,
    w_elec: float = 0.27,
    w_kin: float = 0.15,
    sigma_over_kappa_bulk: float = 16.0,
) -> Any | None:
    """Fit 1D GP strictly over compatible sweep runs (never mixing unrelated presets onto kPa)."""
    items = list(history or [])
    if not items:
        return None

    # Primary campaign: Carbon Felt / PorousDiff clamping-pressure sweep [kPa]
    felt_pts = [
        h for h in items if h.get("source_kind") in ("carbon_felt", "porousdiff_gallery")
    ]
    if active_source_kind in (None, "carbon_felt", "porousdiff_gallery") and len(felt_pts) >= 2:
        xs = [float(p["clamping_kpa"]) for p in felt_pts]
        ys = [
            _recompute_entry_r_cell(
                p,
                w_ion=w_ion,
                w_elec=w_elec,
                w_kin=w_kin,
                sigma_over_kappa_bulk=sigma_over_kappa_bulk,
            )
            for p in felt_pts
        ]
        if len(set(round(x, 2) for x in xs)) >= 2:
            return suggest_next_setpoint_gp(
                xs,
                ys,
                bounds=(20.0, 500.0),
                parameter_name="Clamping pressure",
                parameter_unit="kPa",
            )

    # Secondary campaign: Porosity sweep within a single synthetic generator family
    if active_source_kind:
        same_kind = [h for h in items if h.get("source_kind") == active_source_kind]
        if len(same_kind) >= 2:
            xs = [float(p["porosity"]) for p in same_kind]
            if len(set(round(x, 4) for x in xs)) >= 2:
                ys = [
                    _recompute_entry_r_cell(
                        p,
                        w_ion=w_ion,
                        w_elec=w_elec,
                        w_kin=w_kin,
                        sigma_over_kappa_bulk=sigma_over_kappa_bulk,
                    )
                    for p in same_kind
                ]
                return suggest_next_setpoint_gp(
                    xs,
                    ys,
                    bounds=(0.20, 0.88),
                    parameter_name="Porosity",
                    parameter_unit="-",
                )

    # Fallback to felt campaign if >= 2 felt points exist even while viewing another preset
    if len(felt_pts) >= 2:
        xs = [float(p["clamping_kpa"]) for p in felt_pts]
        ys = [
            _recompute_entry_r_cell(
                p,
                w_ion=w_ion,
                w_elec=w_elec,
                w_kin=w_kin,
                sigma_over_kappa_bulk=sigma_over_kappa_bulk,
            )
            for p in felt_pts
        ]
        if len(set(round(x, 2) for x in xs)) >= 2:
            return suggest_next_setpoint_gp(
                xs,
                ys,
                bounds=(20.0, 500.0),
                parameter_name="Clamping pressure",
                parameter_unit="kPa",
            )

    return None


def register_callbacks(app: Dash, registry: RunRegistry) -> None:
    """Attach all interactive callbacks to `app`."""

    @app.callback(
        Output("theme-store", "data"),
        Input("theme-toggle-btn", "n_clicks"),
        State("theme-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_theme(n_clicks: int, current: str | None) -> str:
        if not n_clicks:
            raise PreventUpdate
        return "dark" if (current or "light") == "light" else "light"

    @app.callback(
        Output("hub-theme-wrapper", "data-theme"),
        Input("theme-store", "data"),
    )
    def apply_theme_attr(theme: str | None) -> str:
        return "dark" if theme == "dark" else "light"

    @app.callback(
        Output("upload-container", "style"),
        Output("grid-side-block", "style"),
        Output("clamping-control-block", "style"),
        Output("porosity-control-block", "style"),
        Output("seed-control-block", "style"),
        Output("aniso-spacing-details", "open"),
        Input("source-kind-dropdown", "value"),
    )
    def update_sidebar_visibility(
        source_kind: str,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str], dict[str, str], bool]:
        is_upload = source_kind == "user_upload"
        is_fixed = source_kind in ("user_upload", "benchmark_dataset", "porousdiff_gallery")
        show_upload = {"display": "block"} if is_upload else {"display": "none"}
        show_grid = {"display": "none"} if is_fixed else {"display": "block"}
        show_clamp = (
            {"display": "block"}
            if source_kind in ("carbon_felt", "porousdiff_gallery")
            else {"display": "none"}
        )
        show_porosity = (
            {"display": "none"}
            if source_kind in ("user_upload", "benchmark_dataset", "open_cube", "porousdiff_gallery")
            else {"display": "block"}
        )
        show_seed = {"display": "none"} if is_fixed else {"display": "block"}
        return show_upload, show_grid, show_clamp, show_porosity, show_seed, is_upload

    @app.callback(
        Output("uploaded-file-store", "data"),
        Output("upload-status-text", "children"),
        Output("upload-seg-drawer", "style"),
        Output("upload-seg-mode-dropdown", "value"),
        Output("upload-threshold-input", "value"),
        Input("upload-volume-box", "contents"),
        State("upload-volume-box", "filename"),
        prevent_initial_call=True,
    )
    def stage_uploaded_volume(
        contents: str | None,
        filename: str | None,
    ) -> tuple[Any, Any, dict[str, str], str, float]:
        if not contents or not filename:
            raise PreventUpdate
        try:
            _, b64_payload = contents.split(",", 1)
            raw_bytes = base64.b64decode(b64_payload)
            staged_path = stage_upload_bytes(raw_bytes, filename)
            info = inspect_staged_volume(staged_path, display_name=filename)
        except Exception as exc:
            err_msg = html.Span(str(exc), style={"color": "#dc2626", "fontWeight": "600"})
            return None, err_msg, {"display": "none"}, "otsu", 0.5

        shp = info["shape_zyx"]
        cal_um = info.get("calibrated_voxel_um_zyx")
        cal_badge = (
            f"Calibrated ({cal_um[0]:g}, {cal_um[1]:g}, {cal_um[2]:g}) µm"
            if cal_um
            else "Uncalibrated — verify (dz, dy, dx) µm"
        )
        default_mode = (
            "labels"
            if (info.get("has_embedded_labels") or info.get("is_integer_labels"))
            else "otsu"
        )
        status = html.Div(
            [
                html.Strong(f"{info['name']} "),
                html.Span(f"({shp[2]}×{shp[1]}×{shp[0]}, {info['dtype']}) · {cal_badge}"),
            ]
        )
        return (
            info,
            status,
            {"display": "flex"},
            default_mode,
            float(info.get("otsu_threshold", 0.5)),
        )

    @app.callback(
        Output("voxel-dz-input", "value"),
        Output("voxel-dy-input", "value"),
        Output("voxel-dx-input", "value"),
        Input("voxel-size-input", "value"),
        Input("uploaded-file-store", "data"),
        prevent_initial_call=True,
    )
    def sync_anisotropic_spacing_inputs(
        iso_um: float | None,
        uploaded_info: dict[str, Any] | None,
    ) -> tuple[Any, Any, Any]:
        trig = ctx.triggered_id
        if trig == "uploaded-file-store" and uploaded_info:
            cal = uploaded_info.get("calibrated_voxel_um_zyx")
            if cal and len(cal) == 3:
                return float(cal[0]), float(cal[1]), float(cal[2])
        if iso_um is not None and float(iso_um) > 0:
            v = float(iso_um)
            return v, v, v
        raise PreventUpdate

    @app.callback(
        Output("upload-preview-graph", "figure"),
        Output("upload-manual-threshold-block", "style"),
        Input("uploaded-file-store", "data"),
        Input("upload-seg-mode-dropdown", "value"),
        Input("upload-threshold-input", "value"),
        Input("upload-polarity-radio", "value"),
        Input("theme-store", "data"),
    )
    def render_upload_segmentation_preview(
        uploaded_info: dict[str, Any] | None,
        seg_mode: str,
        manual_thr: float | None,
        polarity: str,
        theme: str | None,
    ) -> tuple[Any, dict[str, str]]:
        show_manual = {"display": "block"} if seg_mode == "manual" else {"display": "none"}
        dark = theme == "dark"
        if not uploaded_info:
            return (
                empty_placeholder_figure(
                    "Drop a .tif, .npy, .h5, or .ome.zarr.zip above",
                    height=220,
                    dark_mode=dark,
                ),
                show_manual,
            )
        fig = build_upload_segmentation_preview_figure(
            uploaded_info,
            seg_mode=seg_mode or "otsu",
            manual_threshold=manual_thr,
            dark_is_pore=(polarity != "bright"),
            dark_mode=dark,
        )
        return fig, show_manual

    @app.callback(
        Output("active-run-id-store", "data"),
        Output("run-history-store", "data"),
        Output("source-kind-dropdown", "value"),
        Output("grid-side-slider", "value"),
        Output("voxel-size-input", "value"),
        Output("clamping-kpa-slider", "value"),
        Output("porosity-slider", "value"),
        Output("conducting-phase-dropdown", "value"),
        Output("electrode-role-dropdown", "value"),
        Output("homogenisation-check", "value"),
        Output("run-error-banner", "children"),
        Input("run-pipeline-btn", "n_clicks"),
        Input("preset-btn-carbon", "n_clicks"),
        Input("preset-btn-cathode", "n_clicks"),
        Input("preset-btn-cube", "n_clicks"),
        Input("preset-btn-spheres", "n_clicks"),
        Input("run-suggested-setpoint-btn", "n_clicks"),
        Input("reset-campaign-btn", "n_clicks"),
        State("source-kind-dropdown", "value"),
        State("grid-side-slider", "value"),
        State("voxel-size-input", "value"),
        State("voxel-dz-input", "value"),
        State("voxel-dy-input", "value"),
        State("voxel-dx-input", "value"),
        State("clamping-kpa-slider", "value"),
        State("porosity-slider", "value"),
        State("seed-input", "value"),
        State("conducting-phase-dropdown", "value"),
        State("electrode-role-dropdown", "value"),
        State("solver-backend-dropdown", "value"),
        State("homogenisation-check", "value"),
        State("uploaded-file-store", "data"),
        State("upload-seg-mode-dropdown", "value"),
        State("upload-threshold-input", "value"),
        State("upload-polarity-radio", "value"),
        State("upload-coarsen-dropdown", "value"),
        State("suggested-kpa-store", "data"),
        State("run-history-store", "data"),
        State("twin-w-ion-slider", "value"),
        State("twin-w-elec-slider", "value"),
        State("twin-w-kin-slider", "value"),
        State("twin-sigma-kappa-slider", "value"),
        prevent_initial_call=True,
    )
    def execute_run_or_preset(
        _n_run: int,
        _n_p1: int,
        _n_p2: int,
        _n_p3: int,
        _n_p4: int,
        _n_sug: int,
        _n_reset: int,
        source_kind: str,
        grid_side: int,
        voxel_um: float | None,
        voxel_dz: float | None,
        voxel_dy: float | None,
        voxel_dx: float | None,
        clamping_kpa: float,
        porosity: float,
        seed: int | None,
        conducting_phase: int | None,
        electrode_role: str | None,
        solver_backend: str,
        check_flags: list[str] | None,
        uploaded_info: dict[str, Any] | None,
        seg_mode: str | None,
        manual_thr: float | None,
        polarity: str | None,
        coarsen_policy: str | None,
        suggested_val: float | None,
        history: list[dict[str, Any]] | None,
        w_ion: float | None,
        w_elec: float | None,
        w_kin: float | None,
        sigma_kappa: float | None,
    ) -> tuple[Any, ...]:
        trig = ctx.triggered_id
        if trig is None:
            raise PreventUpdate

        if trig == "reset-campaign-btn":
            fresh_hist = initial_history_records(registry)
            def_id = registry.default_run_id or (
                fresh_hist[0]["run_id"] if fresh_hist else no_update
            )
            return (
                def_id,
                fresh_hist,
                "carbon_felt",
                48,
                1.6,
                150.0,
                0.84,
                0,
                "negative",
                [],
                None,
            )

        preset_map = {
            "preset-btn-carbon": "carbon_felt_compression",
            "preset-btn-cathode": "composite_cathode_3phase",
            "preset-btn-cube": "teaching_analytical_cube",
            "preset-btn-spheres": "teaching_sphere_pack",
        }
        weights = (
            float(w_ion if w_ion is not None else 0.58),
            float(w_elec if w_elec is not None else 0.27),
            float(w_kin if w_kin is not None else 0.15),
        )
        sig_kap = float(sigma_kappa if sigma_kappa is not None else 16.0)

        try:
            if trig in preset_map:
                preset = PRESETS[preset_map[trig]]
                side = min(int(preset.microstructure.shape[0]), 64)
                mspec = MicrostructureSpec(
                    **{**preset.microstructure.to_dict(), "shape": (side, side, side)}
                )
                sspec = SolveSpec(
                    **{
                        **preset.solve.to_dict(),
                        "r_cell_weights": weights,
                        "sigma_over_kappa_bulk": sig_kap,
                    }
                )
            elif trig == "run-suggested-setpoint-btn":
                target_kpa = float(suggested_val if suggested_val is not None else 220.0)
                side = int(grid_side or 48)
                mspec = make_carbon_felt_spec(
                    target_kpa,
                    shape=(side, side, side),
                    seed=int(seed if seed is not None else 42),
                )
                base_solve = PRESETS["carbon_felt_compression"].solve
                sspec = SolveSpec(
                    **{
                        **base_solve.to_dict(),
                        "r_cell_weights": weights,
                        "sigma_over_kappa_bulk": sig_kap,
                    }
                )
            else:
                side = int(grid_side or 48)
                sd = int(seed if seed is not None else 42)
                role = str(electrode_role or "negative")
                if source_kind == "multiphase_electrode" and trig == "run-pipeline-btn" and role == "negative":
                    # Preserve user choice if explicitly set, otherwise cathode defaults nicely on Preset #2
                    pass

                # Resolve anisotropic (dz, dy, dx) vs isotropic voxel_size_um
                vz: tuple[float, float, float] | None
                if (
                    voxel_dz is not None
                    and voxel_dy is not None
                    and voxel_dx is not None
                    and float(voxel_dz) > 0
                    and float(voxel_dy) > 0
                    and float(voxel_dx) > 0
                ):
                    vz = (float(voxel_dz), float(voxel_dy), float(voxel_dx))
                elif voxel_um is not None and float(voxel_um) > 0:
                    vz = (float(voxel_um), float(voxel_um), float(voxel_um))
                else:
                    vz = None

                if source_kind == "carbon_felt":
                    base_felt = make_carbon_felt_spec(
                        float(clamping_kpa),
                        shape=(side, side, side),
                        seed=sd,
                    )
                    mspec = MicrostructureSpec(
                        **{
                            **base_felt.to_dict(),
                            "voxel_size_um": vz,
                            "porosity": float(porosity),
                            "electrode_role": role,
                        }
                    )
                elif source_kind == "user_upload":
                    if not uploaded_info or not uploaded_info.get("path"):
                        raise ValueError(
                            "Upload a .tif, .npy, .h5, or .ome.zarr.zip volume before running."
                        )
                    mode = str(seg_mode or "otsu").lower()
                    if mode == "manual":
                        thr_arg: float | str | None = float(
                            manual_thr
                            if manual_thr is not None
                            else uploaded_info.get("otsu_threshold", 0.5)
                        )
                    elif mode == "labels":
                        thr_arg = "labels"
                    else:
                        thr_arg = "otsu"

                    cal_orig = uploaded_info.get("calibrated_voxel_um_zyx")
                    override_cal = False
                    if cal_orig and vz is not None:
                        override_cal = any(
                            abs(float(vz[i]) - float(cal_orig[i])) > 1e-4 for i in range(3)
                        )

                    mspec = MicrostructureSpec(
                        source_kind="user_upload",
                        label=f"Upload: {uploaded_info['name']}",
                        shape=(side, side, side),
                        voxel_size_um=vz,
                        seed=sd,
                        upload_path=str(uploaded_info["path"]),
                        threshold=thr_arg,
                        dark_is_pore=(polarity != "bright"),
                        override_calibrated_spacing=override_cal,
                        coarsen_policy=(
                            "mode_downsample"
                            if coarsen_policy == "mode_downsample"
                            else "crop_center"
                        ),
                        electrode_role=role,  # type: ignore[arg-type]
                    )
                else:
                    mspec = MicrostructureSpec(
                        source_kind=source_kind,  # type: ignore[arg-type]
                        label=f"{source_kind} ({side}³)",
                        shape=(side, side, side),
                        voxel_size_um=vz,
                        seed=sd,
                        porosity=float(porosity),
                        clamping_kpa=float(clamping_kpa),
                        dataset_name=(
                            "sphere_pack" if source_kind == "benchmark_dataset" else None
                        ),
                        electrode_role=role,  # type: ignore[arg-type]
                    )

                cond_id = int(conducting_phase if conducting_phase is not None else 0)
                phase_diffs = (
                    {0: 1.0, 1: 0.0, 2: 0.18}
                    if (mspec.source_kind == "multiphase_electrode" and cond_id == 0)
                    else None
                )
                if mspec.source_kind == "open_cube":
                    solid_p: tuple[int, ...] = ()
                elif cond_id == 1:
                    solid_p = (0,)
                else:
                    solid_p = (1, 2)

                flags = set(check_flags or [])
                sspec = SolveSpec(
                    conducting_phase=cond_id,
                    solid_phases=solid_p,
                    phase_diffusivities=phase_diffs,
                    directions=("x", "y", "z"),
                    solver=solver_backend,  # type: ignore[arg-type]
                    compute_homogenisation=("homog" in flags),
                    compute_solid_wiring=(mspec.source_kind != "open_cube"),
                    compute_rev_study=("rev" in flags),
                    return_fields=True,
                    field_direction="z",
                    r_cell_weights=weights,
                    sigma_over_kappa_bulk=sig_kap,
                )

            executed = run_transport_pipeline(mspec, sspec)
            rid = registry.put(executed)
            new_hist = list(history or [])
            new_hist.append(_history_entry_from_run(executed))

            vz_out = executed.summary.voxel_size_um_zyx[2]
            checks_out = []
            if sspec.compute_homogenisation:
                checks_out.append("homog")
            if sspec.compute_rev_study:
                checks_out.append("rev")

            return (
                rid,
                new_hist,
                mspec.source_kind,
                min(96, max(32, int(executed.phase_volume.shape[0]))),
                vz_out,
                float(mspec.clamping_kpa),
                float(mspec.porosity),
                int(sspec.conducting_phase),
                mspec.electrode_role,
                checks_out,
                None,
            )
        except Exception as exc:
            err_div = html.Div(str(exc), className="error-box")
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                err_div,
            )

    @app.callback(
        Output("main-tabs", "value"),
        Output("slice-plane-radio", "value"),
        Output("slice-layer-radio", "value"),
        Input("jump-roi-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def jump_to_bottleneck_roi(n_clicks: int) -> tuple[str, str, str]:
        if not n_clicks:
            raise PreventUpdate
        return "tab-2d", "xz", "flux"

    @app.callback(
        Output("slice-index-slider", "max"),
        Output("slice-index-slider", "value"),
        Input("active-run-id-store", "data"),
        Input("slice-plane-radio", "value"),
        Input("jump-roi-btn", "n_clicks"),
    )
    def sync_slice_slider_bounds(
        run_id: str | None,
        plane: str,
        _n_jump: int,
    ) -> tuple[int, int]:
        run = registry.get(run_id)
        if run is None:
            return 47, 24
        max_idx, mid_idx = plane_slider_bounds_for_run(run, plane)
        return max(1, max_idx), mid_idx

    @app.callback(
        Output("preset-context-banner", "children"),
        Output("kpi-cards-row", "children"),
        Output("pybamm-bpx-preview", "children"),
        Input("active-run-id-store", "data"),
    )
    def render_summary_strip(run_id: str | None) -> tuple[Any, list[Any], str]:
        run = registry.get(run_id)
        if run is None:
            return html.Div("No active run."), [], "{}"

        s = run.summary
        tz = s.directional.get("z")
        tx = s.directional.get("x")
        ty = s.directional.get("y")
        tau_z_str = f"{tz.tortuosity:.3f}" if (tz and tz.tortuosity is not None) else "No perc."
        deff_z_str = f"{tz.effective_diffusivity:.3f}" if tz else "0.000"
        inplane_tau = (
            0.5 * ((tx.tortuosity or 0.0) + (ty.tortuosity or 0.0))
            if (tx and ty and tx.tortuosity and ty.tortuosity)
            else None
        )
        aniso_str = (
            f"τ_xy = {inplane_tau:.3f} ({s.anisotropy_ratio_z_over_xy:.2f}× Z/XY)"
            if (inplane_tau and s.anisotropy_ratio_z_over_xy)
            else "Isotropic / 1D"
        )

        vz = s.voxel_size_um_zyx
        vox_badge = (
            f"{vz[0]:g} µm"
            if (abs(vz[0] - vz[1]) < 1e-4 and abs(vz[0] - vz[2]) < 1e-4)
            else f"({vz[0]:g}, {vz[1]:g}, {vz[2]:g}) µm"
        )

        banner = [
            html.Div(
                children=[
                    html.Span(f"origin: {s.data_origin}", className="pill pill-origin"),
                    html.Span(f"role: {s.electrode_role}", className="pill"),
                    html.Span(f"solver: {s.solver_backend}", className="pill"),
                    html.Span(
                        f"{s.shape_zyx[2]}×{s.shape_zyx[1]}×{s.shape_zyx[0]} @ {vox_badge}",
                        className="pill",
                    ),
                    html.Strong(s.label, style={"fontSize": "15px", "marginLeft": "4px"}),
                    html.Div(
                        f"{s.precision_tag}"
                        + (f" · {s.coarsened_note}" if s.coarsened_note else ""),
                        style={
                            "fontSize": "12px",
                            "color": "var(--text-muted)",
                            "marginTop": "3px",
                        },
                    ),
                ]
            ),
            html.Div(
                f"Gen {s.generation_time_s*1000:.0f} ms · Solve {s.solve_time_s:.2f} s · OpenImpala v{s.openimpala_version}",
                style={
                    "fontSize": "12px",
                    "fontWeight": "600",
                    "color": "var(--text-secondary)",
                },
            ),
        ]

        phase_label = (
            f"Conducting Phase ({s.conducting_phase}) ε"
            if s.conducting_phase != 0
            else "Conducting Porosity ε"
        )
        cards = [
            html.Div(
                className="kpi-card",
                children=[
                    html.Div(phase_label, className="kpi-label"),
                    html.Div(f"{s.porosity:.4f}", className="kpi-value"),
                    html.Div(f"{s.porosity*100:.1f}% volume fraction", className="kpi-sub"),
                ],
            ),
            html.Div(
                className="kpi-card",
                children=[
                    html.Div("Through-Plane Tortuosity τ_Z", className="kpi-label"),
                    html.Div(tau_z_str, className="kpi-value"),
                    html.Div(aniso_str, className="kpi-sub"),
                ],
            ),
            html.Div(
                className="kpi-card",
                children=[
                    html.Div("Effective Diffusivity D_eff,Z / D_0", className="kpi-label"),
                    html.Div(deff_z_str, className="kpi-value"),
                    html.Div(
                        f"MacMullin N_M = {tz.macmullin_number:.2f}"
                        if (tz and tz.macmullin_number)
                        else "Unpercolated",
                        className="kpi-sub",
                    ),
                ],
            ),
            html.Div(
                className="kpi-card",
                children=[
                    html.Div("Specific Surface Area (2/3 CC)", className="kpi-label"),
                    html.Div(f"{s.specific_surface_area_m_inv:,.0f}", className="kpi-value"),
                    html.Div("m⁻¹ (Cauchy–Crofton unbiased)", className="kpi-sub"),
                ],
            ),
            html.Div(
                className="kpi-card",
                children=[
                    html.Div("Solid Z-Conductivity σ_eff,Z / σ_0", className="kpi-label"),
                    html.Div(f"{s.solid_effective_conductivity_z:.3f}", className="kpi-value"),
                    html.Div(
                        "Through-plane wired"
                        if s.solid_through_plane_percolates
                        else "Single-phase / unpercolated",
                        className="kpi-sub",
                    ),
                ],
            ),
            html.Div(
                className="kpi-card",
                children=[
                    html.Div("Series Cell Resistance R_cell", className="kpi-label"),
                    html.Div(f"{s.cell_resistance_fom:.3f}", className="kpi-value"),
                    html.Div("Digital-twin objective (lower = better)", className="kpi-sub"),
                ],
            ),
        ]

        readout: dict[str, Any] = {
            "electrode_role": s.electrode_role,
            "pybamm_parameters": s.pybamm_parameters,
            "homogenised_tensor_xyz": s.homogenised_tensor_xyz,
            "phase_fractions": s.phase_fractions,
        }
        if s.rev_sizes_vox:
            readout["rev_study"] = {
                "sizes_vox": s.rev_sizes_vox,
                "deff_mean": s.rev_deff_mean,
                "deff_std": s.rev_deff_std,
            }
        return banner, cards, json.dumps(readout, indent=2)

    @app.callback(
        Output("slice-heatmap-graph", "figure"),
        Output("directional-profile-graph", "figure"),
        Input("active-run-id-store", "data"),
        Input("slice-plane-radio", "value"),
        Input("slice-layer-radio", "value"),
        Input("slice-index-slider", "value"),
        Input("theme-store", "data"),
    )
    def render_2d_and_profile(
        run_id: str | None,
        plane: str,
        layer: str,
        slice_idx: int,
        theme: str | None,
    ) -> tuple[Any, Any]:
        dark = theme == "dark"
        run = registry.get(run_id)
        if run is None:
            empty = empty_placeholder_figure(dark_mode=dark)
            return empty, empty
        fig_slice = build_slice_figure(
            run,
            plane=plane,  # type: ignore[arg-type]
            slice_index=int(slice_idx or 0),
            layer=layer,  # type: ignore[arg-type]
            dark_mode=dark,
        )
        fig_prof = build_directional_and_profile_figure(run, dark_mode=dark)
        return fig_slice, fig_prof

    @app.callback(
        Output("isosurface-3d-graph", "figure"),
        Input("main-tabs", "value"),
        Input("active-run-id-store", "data"),
        Input("options-3d-check", "value"),
        Input("opacity-3d-slider", "value"),
        Input("clip-z-slider", "value"),
        Input("clip-x-slider", "value"),
        Input("theme-store", "data"),
    )
    def render_3d_tab_gated(
        active_tab: str,
        run_id: str | None,
        opts: list[str] | None,
        opacity: float,
        clip_z: list[float],
        clip_x: list[float],
        theme: str | None,
    ) -> Any:
        if active_tab != "tab-3d":
            raise PreventUpdate
        dark = theme == "dark"
        run = registry.get(run_id)
        if run is None:
            return empty_placeholder_figure(dark_mode=dark, height=540)
        flags = set(opts or [])
        return build_3d_isosurface_figure(
            run,
            clip_pct_zyx=(
                (float(clip_z[0]), float(clip_z[1])),
                (0.0, 100.0),
                (float(clip_x[0]), float(clip_x[1])),
            ),
            opacity=float(opacity or 0.9),
            show_coating=("coating" in flags),
            show_roi_box=("roi" in flags),
            dark_mode=dark,
        )

    @app.callback(
        Output("compare-left-dropdown", "options"),
        Output("compare-right-dropdown", "options"),
        Output("compare-left-graph", "figure"),
        Output("compare-right-graph", "figure"),
        Output("compare-delta-table", "children"),
        Input("main-tabs", "value"),
        Input("run-history-store", "data"),
        Input("compare-left-dropdown", "value"),
        Input("compare-right-dropdown", "value"),
        Input("compare-layer-radio", "value"),
        Input("theme-store", "data"),
    )
    def render_compare_tab(
        active_tab: str,
        history: list[dict[str, Any]] | None,
        left_id: str | None,
        right_id: str | None,
        layer: str,
        theme: str | None,
    ) -> tuple[Any, ...]:
        opts = [
            {"label": f"{h['label']} (R={h['r_cell']:.2f})", "value": h["run_id"]}
            for h in (history or [])
        ]
        if active_tab != "tab-compare":
            return opts, opts, no_update, no_update, no_update

        dark = theme == "dark"
        r_left = registry.get(left_id) or (registry.get(opts[0]["value"]) if opts else None)
        r_right = registry.get(right_id) or (registry.get(opts[-1]["value"]) if opts else None)
        if r_left is None or r_right is None:
            emp = empty_placeholder_figure(dark_mode=dark)
            return opts, opts, emp, emp, html.Div()

        shared_zmax: float | None = None
        if layer == "flux" and r_left.flux_field is not None and r_right.flux_field is not None:
            shared_zmax = max(
                float(np.percentile(r_left.flux_field, 99.2)),
                float(np.percentile(r_right.flux_field, 99.2)),
                1.25,
            )

        fig_l = build_slice_figure(
            r_left,
            plane="xz",
            slice_index=r_left.phase_volume.shape[1] // 2,
            layer=layer,  # type: ignore[arg-type]
            zmax_override=shared_zmax,
            height=410,
            dark_mode=dark,
        )
        fig_r = build_slice_figure(
            r_right,
            plane="xz",
            slice_index=r_right.phase_volume.shape[1] // 2,
            layer=layer,  # type: ignore[arg-type]
            zmax_override=shared_zmax,
            height=410,
            dark_mode=dark,
        )

        sl, sr = r_left.summary, r_right.summary
        tz_l = sl.directional["z"].tortuosity if "z" in sl.directional else None
        tz_r = sr.directional["z"].tortuosity if "z" in sr.directional else None
        delta_tbl = html.Table(
            className="roi-table",
            children=[
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Metric"),
                            html.Th(f"Left: {sl.label}"),
                            html.Th(f"Right: {sr.label}"),
                            html.Th("Delta (Right − Left)"),
                        ]
                    )
                ),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td("Porosity ε"),
                                html.Td(f"{sl.porosity:.4f}"),
                                html.Td(f"{sr.porosity:.4f}"),
                                html.Td(f"{sr.porosity - sl.porosity:+.4f}"),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td("Through-Plane τ_Z"),
                                html.Td(f"{tz_l:.3f}" if tz_l else "N/A"),
                                html.Td(f"{tz_r:.3f}" if tz_r else "N/A"),
                                html.Td(f"{tz_r - tz_l:+.3f}" if (tz_l and tz_r) else "—"),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td("Solid Z-Conductivity σ_eff,Z"),
                                html.Td(f"{sl.solid_effective_conductivity_z:.4f}"),
                                html.Td(f"{sr.solid_effective_conductivity_z:.4f}"),
                                html.Td(
                                    f"{sr.solid_effective_conductivity_z - sl.solid_effective_conductivity_z:+.4f}"
                                ),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td("Cell Resistance R_cell"),
                                html.Td(f"{sl.cell_resistance_fom:.3f}"),
                                html.Td(f"{sr.cell_resistance_fom:.3f}"),
                                html.Td(f"{sr.cell_resistance_fom - sl.cell_resistance_fom:+.3f}"),
                            ]
                        ),
                    ]
                ),
            ],
        )
        return opts, opts, fig_l, fig_r, delta_tbl

    @app.callback(
        Output("gp-steering-graph", "figure"),
        Output("twin-recommendation-banner", "children"),
        Output("suggested-kpa-store", "data"),
        Output("zoom-roi-cards", "children"),
        Output("sweep-history-table", "children"),
        Input("active-run-id-store", "data"),
        Input("run-history-store", "data"),
        Input("twin-w-ion-slider", "value"),
        Input("twin-w-elec-slider", "value"),
        Input("twin-w-kin-slider", "value"),
        Input("twin-sigma-kappa-slider", "value"),
        Input("theme-store", "data"),
    )
    def render_digital_twin_tab(
        run_id: str | None,
        history: list[dict[str, Any]] | None,
        w_ion: float | None,
        w_elec: float | None,
        w_kin: float | None,
        sigma_kappa: float | None,
        theme: str | None,
    ) -> tuple[Any, ...]:
        dark = theme == "dark"
        run = registry.get(run_id)
        active_kind = run.summary.source_kind if run else None

        wi = float(w_ion if w_ion is not None else 0.58)
        we = float(w_elec if w_elec is not None else 0.27)
        wk = float(w_kin if w_kin is not None else 0.15)
        sk = float(sigma_kappa if sigma_kappa is not None else 16.0)

        sug = compute_gp_for_campaign(
            history or [],
            active_source_kind=active_kind,
            w_ion=wi,
            w_elec=we,
            w_kin=wk,
            sigma_over_kappa_bulk=sk,
        )

        if sug is not None:
            gp_fig = build_gp_steering_figure(sug, dark_mode=dark)
            rec_banner = html.Div(
                children=[
                    html.Span("Bayesian Active Planner", className="pill pill-origin"),
                    html.Strong(
                        f"Next Recommended Setpoint: {sug.suggested_value:.1f} {sug.parameter_unit}".strip(),
                        style={"fontSize": "14.5px"},
                    ),
                    html.Div(
                        sug.rationale,
                        style={
                            "fontSize": "12px",
                            "color": "var(--text-secondary)",
                            "marginTop": "3px",
                        },
                    ),
                ]
            )
            next_kpa = sug.suggested_value if sug.parameter_unit == "kPa" else 200.0
        else:
            gp_fig = empty_placeholder_figure(
                "Run 2+ compression (or porosity) states of the same family to fit the GP surrogate.",
                dark_mode=dark,
            )
            rec_banner = html.Div(
                "Evaluate at least two distinct compression or porosity states to activate GP steering."
            )
            next_kpa = 200.0

        roi_rows = []
        if run and run.summary.zoom_rois:
            for r in run.summary.zoom_rois:
                cz, cy, cx = r.center_um_zyx
                roi_rows.append(
                    html.Tr(
                        [
                            html.Td(f"#{r.rank}"),
                            html.Td(f"({cx:.1f}, {cy:.1f}, {cz:.1f}) µm"),
                            html.Td(f"{r.peak_flux_magnitude:.2f}×"),
                            html.Td(f"{r.local_porosity:.2f} ({r.constriction_index:.2f}×)"),
                        ]
                    )
                )
        roi_block = (
            html.Table(
                className="roi-table",
                children=[
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("ROI"),
                                html.Th("Centre (x, y, z)"),
                                html.Th("Peak |J/J0|"),
                                html.Th("Local ε (Constr.)"),
                            ]
                        )
                    ),
                    html.Tbody(roi_rows),
                ],
            )
            if roi_rows
            else html.Div(
                "Uniform domain — no localised bottleneck throats.",
                style={"fontSize": "12.5px"},
            )
        )

        hist_rows = []
        for h in history or []:
            live_r = _recompute_entry_r_cell(
                h, w_ion=wi, w_elec=we, w_kin=wk, sigma_over_kappa_bulk=sk
            )
            param_str = (
                f"{h['clamping_kpa']:.0f} kPa"
                if h.get("source_kind") in ("carbon_felt", "porousdiff_gallery")
                else f"ε={h['porosity']:.2f}"
            )
            hist_rows.append(
                html.Tr(
                    [
                        html.Td(h["label"]),
                        html.Td(param_str),
                        html.Td(f"{h['porosity']:.3f}"),
                        html.Td(f"{h['tau_z']:.3f}" if h.get("tau_z") else "—"),
                        html.Td(html.B(f"{live_r:.3f}")),
                    ]
                )
            )
        hist_tbl = html.Table(
            className="roi-table",
            children=[
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Run"),
                            html.Th("Setpoint"),
                            html.Th("ε"),
                            html.Th("τ_Z"),
                            html.Th("R_cell (Live)"),
                        ]
                    )
                ),
                html.Tbody(hist_rows),
            ],
        )
        return gp_fig, rec_banner, next_kpa, roi_block, hist_tbl

    @app.callback(
        Output("download-bundle", "data"),
        Input("export-bundle-btn", "n_clicks"),
        State("active-run-id-store", "data"),
        State("run-history-store", "data"),
        State("twin-w-ion-slider", "value"),
        State("twin-w-elec-slider", "value"),
        State("twin-w-kin-slider", "value"),
        State("twin-sigma-kappa-slider", "value"),
        prevent_initial_call=True,
    )
    def download_nia_bundle(
        n_clicks: int,
        run_id: str | None,
        history: list[dict[str, Any]] | None,
        w_ion: float | None,
        w_elec: float | None,
        w_kin: float | None,
        sigma_kappa: float | None,
    ) -> Any:
        if not n_clicks:
            raise PreventUpdate
        run = registry.get_or_rehydrate(run_id, history)
        if run is None:
            raise PreventUpdate
        sug = compute_gp_for_campaign(
            history or [],
            active_source_kind=run.summary.source_kind,
            w_ion=float(w_ion if w_ion is not None else 0.58),
            w_elec=float(w_elec if w_elec is not None else 0.27),
            w_kin=float(w_kin if w_kin is not None else 0.15),
            sigma_over_kappa_bulk=float(sigma_kappa if sigma_kappa is not None else 16.0),
        )
        payload = build_report_bundle_bytes(run, bayesian_suggestion=sug, include_ome_zarr=True)
        fname = f"openimpala_hub_bundle_{run.summary.run_id}.zip"
        return dcc.send_bytes(payload, filename=fname)

    @app.callback(
        Output("download-bpx", "data"),
        Input("export-bpx-btn", "n_clicks"),
        State("active-run-id-store", "data"),
        State("run-history-store", "data"),
        prevent_initial_call=True,
    )
    def download_bpx_json(
        n_clicks: int,
        run_id: str | None,
        history: list[dict[str, Any]] | None,
    ) -> Any:
        if not n_clicks:
            raise PreventUpdate
        run = registry.get_or_rehydrate(run_id, history)
        if run is None:
            raise PreventUpdate
        content = json.dumps(run.summary.bpx_parameters, indent=2)
        fname = f"openimpala_bpx_{run.summary.run_id}.json"
        return dcc.send_string(content, filename=fname)
