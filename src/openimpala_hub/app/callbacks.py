"""Reactive Dash callbacks connecting the UI to the headless OpenImpala-Hub engine."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import tempfile
from typing import Any

from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
import numpy as np

from ..contracts import MicrostructureSpec, SolveSpec
from ..engine.bundle import build_report_bundle_bytes
from ..engine.solver import ExecutedRun, run_transport_pipeline
from ..engine.steering import suggest_next_setpoint_gp
from ..presets import PRESETS, make_carbon_felt_spec
from .figures import (
    build_3d_isosurface_figure,
    build_directional_and_profile_figure,
    build_gp_steering_figure,
    build_slice_figure,
    empty_placeholder_figure,
)
from .store import RunRegistry

_UPLOAD_STAGE_DIR = Path(tempfile.gettempdir()) / "openimpala_hub_uploads"
_MAX_UPLOAD_BYTES = 64 * 1024 * 1024  # 64 MiB cap on public host


def _history_entry_from_run(run: ExecutedRun) -> dict[str, Any]:
    s = run.summary
    tz = s.directional.get("z")
    return {
        "run_id": s.run_id,
        "label": s.label,
        "clamping_kpa": float(s.microstructure_spec.get("clamping_kpa", 150.0)),
        "porosity": s.porosity,
        "tau_z": tz.tortuosity if tz else None,
        "d_eff_z": tz.effective_diffusivity if tz else 0.0,
        "sigma_z": s.solid_effective_conductivity_z,
        "r_cell": s.cell_resistance_fom,
        "source_kind": s.source_kind,
    }


def _compute_gp_from_history(history: list[dict[str, Any]]) -> Any | None:
    felt_pts = [
        h for h in (history or []) if h.get("source_kind") in ("carbon_felt", "porousdiff_gallery")
    ]
    if len(felt_pts) < 2:
        felt_pts = list(history or [])
    if not felt_pts:
        return None
    xs = [float(p["clamping_kpa"]) for p in felt_pts]
    ys = [float(p["r_cell"]) for p in felt_pts]
    return suggest_next_setpoint_gp(xs, ys, bounds=(20.0, 500.0))


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
        Output("clamping-control-block", "style"),
        Input("source-kind-dropdown", "value"),
    )
    def update_sidebar_visibility(source_kind: str) -> tuple[dict[str, str], dict[str, str]]:
        show_upload = {"display": "block"} if source_kind == "user_upload" else {"display": "none"}
        show_clamp = (
            {"display": "block"}
            if source_kind in ("carbon_felt", "porousdiff_gallery")
            else {"display": "none"}
        )
        return show_upload, show_clamp

    @app.callback(
        Output("uploaded-file-store", "data"),
        Output("upload-status-text", "children"),
        Input("upload-volume-box", "contents"),
        State("upload-volume-box", "filename"),
        prevent_initial_call=True,
    )
    def stage_uploaded_volume(contents: str | None, filename: str | None) -> tuple[Any, str]:
        if not contents or not filename:
            raise PreventUpdate
        safe_name = Path(filename).name
        if not safe_name.lower().endswith((".tif", ".tiff", ".npy", ".h5", ".hdf5")):
            return None, f"Unsupported file '{safe_name}'. Upload .tif, .npy, or .h5."
        _, b64_payload = contents.split(",", 1)
        raw_bytes = base64.b64decode(b64_payload)
        if len(raw_bytes) > _MAX_UPLOAD_BYTES:
            return None, f"File exceeds 64 MiB public upload limit ({len(raw_bytes) // (1024*1024)} MiB)."
        _UPLOAD_STAGE_DIR.mkdir(parents=True, exist_ok=True)
        dest = _UPLOAD_STAGE_DIR / safe_name
        dest.write_bytes(raw_bytes)
        return {"path": str(dest), "name": safe_name}, f"Staged: {safe_name} ({len(raw_bytes) // 1024:,} KiB)"

    @app.callback(
        Output("active-run-id-store", "data"),
        Output("run-history-store", "data"),
        Output("source-kind-dropdown", "value"),
        Output("grid-side-slider", "value"),
        Output("voxel-size-input", "value"),
        Output("clamping-kpa-slider", "value"),
        Output("porosity-slider", "value"),
        Output("homogenisation-check", "value"),
        Output("slice-index-slider", "max"),
        Output("slice-index-slider", "value"),
        Output("run-error-banner", "children"),
        Input("run-pipeline-btn", "n_clicks"),
        Input("preset-btn-carbon", "n_clicks"),
        Input("preset-btn-cathode", "n_clicks"),
        Input("preset-btn-cube", "n_clicks"),
        Input("preset-btn-spheres", "n_clicks"),
        Input("run-suggested-setpoint-btn", "n_clicks"),
        State("source-kind-dropdown", "value"),
        State("grid-side-slider", "value"),
        State("voxel-size-input", "value"),
        State("clamping-kpa-slider", "value"),
        State("porosity-slider", "value"),
        State("seed-input", "value"),
        State("solver-backend-dropdown", "value"),
        State("homogenisation-check", "value"),
        State("uploaded-file-store", "data"),
        State("suggested-kpa-store", "data"),
        State("run-history-store", "data"),
        prevent_initial_call=True,
    )
    def execute_run_or_preset(
        _n_run: int,
        _n_p1: int,
        _n_p2: int,
        _n_p3: int,
        _n_p4: int,
        _n_sug: int,
        source_kind: str,
        grid_side: int,
        voxel_um: float | None,
        clamping_kpa: float,
        porosity: float,
        seed: int | None,
        solver_backend: str,
        homog_flags: list[str] | None,
        uploaded_info: dict[str, str] | None,
        suggested_kpa: float | None,
        history: list[dict[str, Any]] | None,
    ) -> tuple[Any, ...]:
        trig = ctx.triggered_id
        if trig is None:
            raise PreventUpdate

        preset_map = {
            "preset-btn-carbon": "carbon_felt_compression",
            "preset-btn-cathode": "composite_cathode_3phase",
            "preset-btn-cube": "teaching_analytical_cube",
            "preset-btn-spheres": "teaching_sphere_pack",
        }

        try:
            if trig in preset_map:
                preset = PRESETS[preset_map[trig]]
                # Use 48^3 on web presets for snappy interactive turnaround (< 1s)
                side = min(int(preset.microstructure.shape[0]), 64)
                mspec = MicrostructureSpec(
                    **{**preset.microstructure.to_dict(), "shape": (side, side, side)}
                )
                sspec = preset.solve
            elif trig == "run-suggested-setpoint-btn":
                target_kpa = float(suggested_kpa if suggested_kpa is not None else 220.0)
                side = int(grid_side or 48)
                mspec = make_carbon_felt_spec(
                    target_kpa,
                    shape=(side, side, side),
                    seed=int(seed if seed is not None else 42),
                )
                sspec = PRESETS["carbon_felt_compression"].solve
            else:
                side = int(grid_side or 48)
                vz = (
                    (float(voxel_um), float(voxel_um), float(voxel_um))
                    if voxel_um is not None and float(voxel_um) > 0
                    else None
                )
                sd = int(seed if seed is not None else 42)
                if source_kind == "carbon_felt":
                    mspec = make_carbon_felt_spec(
                        float(clamping_kpa),
                        shape=(side, side, side),
                        seed=sd,
                    )
                    if vz is not None:
                        mspec = MicrostructureSpec(
                            **{
                                **mspec.to_dict(),
                                "voxel_size_um": vz,
                                "porosity": float(porosity),
                            }
                        )
                    else:
                        mspec = MicrostructureSpec(**{**mspec.to_dict(), "voxel_size_um": None})
                else:
                    up_path = uploaded_info["path"] if uploaded_info else None
                    lbl = (
                        f"Upload: {uploaded_info['name']}"
                        if (source_kind == "user_upload" and uploaded_info)
                        else f"{source_kind} ({side}³)"
                    )
                    mspec = MicrostructureSpec(
                        source_kind=source_kind,  # type: ignore[arg-type]
                        label=lbl,
                        shape=(side, side, side),
                        voxel_size_um=vz,
                        seed=sd,
                        porosity=float(porosity),
                        clamping_kpa=float(clamping_kpa),
                        dataset_name="sphere_pack" if source_kind == "benchmark_dataset" else None,
                        upload_path=up_path,
                        threshold="otsu" if source_kind == "user_upload" else None,
                    )

                phase_diffs = (
                    {0: 1.0, 1: 0.0, 2: 0.18}
                    if mspec.source_kind == "multiphase_electrode"
                    else None
                )
                solid_p = () if mspec.source_kind == "open_cube" else (1, 2)
                sspec = SolveSpec(
                    conducting_phase=0,
                    solid_phases=solid_p,
                    phase_diffusivities=phase_diffs,
                    directions=("x", "y", "z"),
                    solver=solver_backend,  # type: ignore[arg-type]
                    compute_homogenisation=bool(homog_flags and "homog" in homog_flags),
                    compute_solid_wiring=(mspec.source_kind != "open_cube"),
                    return_fields=True,
                    field_direction="z",
                )

            executed = run_transport_pipeline(mspec, sspec)
            rid = registry.put(executed)
            new_hist = list(history or [])
            new_hist.append(_history_entry_from_run(executed))

            ny = executed.phase_volume.shape[1]
            mid_slice = ny // 2
            if executed.summary.zoom_rois:
                (_, _), (y0, y1), (_, _) = executed.summary.zoom_rois[0].bounds_zyx
                mid_slice = (y0 + y1) // 2

            vz_out = executed.summary.voxel_size_um_zyx[0]
            homog_out = ["homog"] if sspec.compute_homogenisation else []
            return (
                rid,
                new_hist,
                mspec.source_kind,
                executed.phase_volume.shape[0],
                vz_out,
                float(mspec.clamping_kpa),
                float(mspec.porosity),
                homog_out,
                max(1, ny - 1),
                mid_slice,
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

        banner = [
            html.Div(
                children=[
                    html.Span(f"origin: {s.data_origin}", className="pill pill-origin"),
                    html.Span(f"solver: {s.solver_backend}", className="pill"),
                    html.Span(
                        f"{s.shape_zyx[2]}×{s.shape_zyx[1]}×{s.shape_zyx[0]} @ {s.voxel_size_um_zyx[0]:g} µm",
                        className="pill",
                    ),
                    html.Strong(s.label, style={"fontSize": "15px", "marginLeft": "4px"}),
                    html.Div(
                        f"{s.precision_tag}"
                        + (f" · {s.coarsened_note}" if s.coarsened_note else ""),
                        style={"fontSize": "12px", "color": "var(--text-muted)", "marginTop": "3px"},
                    ),
                ]
            ),
            html.Div(
                f"Gen {s.generation_time_s*1000:.0f} ms · Solve {s.solve_time_s:.2f} s · OpenImpala v{s.openimpala_version}",
                style={"fontSize": "12px", "fontWeight": "600", "color": "var(--text-secondary)"},
            ),
        ]

        cards = [
            html.Div(
                className="kpi-card",
                children=[
                    html.Div("Conducting Porosity ε", className="kpi-label"),
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
                        "Through-plane wired" if s.solid_through_plane_percolates else "Single-phase / unpercolated",
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

        readout = {
            "pybamm_parameters": s.pybamm_parameters,
            "homogenised_tensor_xyz": s.homogenised_tensor_xyz,
            "phase_fractions": s.phase_fractions,
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
        # Gate marching-cubes so off-screen 2D interactions stay instant
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
        opts = [{"label": f"{h['label']} (R={h['r_cell']:.2f})", "value": h["run_id"]} for h in (history or [])]
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
        Input("theme-store", "data"),
    )
    def render_digital_twin_tab(
        run_id: str | None,
        history: list[dict[str, Any]] | None,
        theme: str | None,
    ) -> tuple[Any, ...]:
        dark = theme == "dark"
        run = registry.get(run_id)
        sug = _compute_gp_from_history(history or [])

        if sug is not None:
            gp_fig = build_gp_steering_figure(sug, dark_mode=dark)
            rec_banner = html.Div(
                children=[
                    html.Span("Bayesian Active Planner", className="pill pill-origin"),
                    html.Strong(
                        f"Next Recommended Setpoint: {sug.suggested_value:.1f} {sug.parameter_unit}",
                        style={"fontSize": "14.5px"},
                    ),
                    html.Div(
                        sug.rationale,
                        style={"fontSize": "12px", "color": "var(--text-secondary)", "marginTop": "3px"},
                    ),
                ]
            )
            next_kpa = sug.suggested_value
        else:
            gp_fig = empty_placeholder_figure(
                "Run 2+ compression states to fit the Gaussian Process surrogate.",
                dark_mode=dark,
            )
            rec_banner = html.Div("Run additional compression states to activate GP steering.")
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
            else html.Div("Uniform domain — no localised bottleneck throats.", style={"fontSize": "12.5px"})
        )

        hist_rows = [
            html.Tr(
                [
                    html.Td(h["label"]),
                    html.Td(f"{h['clamping_kpa']:.0f} kPa"),
                    html.Td(f"{h['porosity']:.3f}"),
                    html.Td(f"{h['tau_z']:.3f}" if h.get("tau_z") else "—"),
                    html.Td(html.B(f"{h['r_cell']:.3f}")),
                ]
            )
            for h in (history or [])
        ]
        hist_tbl = html.Table(
            className="roi-table",
            children=[
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Run"),
                            html.Th("Load"),
                            html.Th("ε"),
                            html.Th("τ_Z"),
                            html.Th("R_cell"),
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
        prevent_initial_call=True,
    )
    def download_nia_bundle(
        n_clicks: int,
        run_id: str | None,
        history: list[dict[str, Any]] | None,
    ) -> Any:
        if not n_clicks:
            raise PreventUpdate
        run = registry.get(run_id)
        if run is None:
            raise PreventUpdate
        sug = _compute_gp_from_history(history or [])
        payload = build_report_bundle_bytes(run, bayesian_suggestion=sug, include_ome_zarr=True)
        fname = f"openimpala_hub_bundle_{run.summary.run_id}.zip"
        return dcc.send_bytes(payload, filename=fname)

    @app.callback(
        Output("download-bpx", "data"),
        Input("export-bpx-btn", "n_clicks"),
        State("active-run-id-store", "data"),
        prevent_initial_call=True,
    )
    def download_bpx_json(n_clicks: int, run_id: str | None) -> Any:
        if not n_clicks:
            raise PreventUpdate
        run = registry.get(run_id)
        if run is None:
            raise PreventUpdate
        content = json.dumps(run.summary.bpx_parameters, indent=2)
        fname = f"openimpala_bpx_{run.summary.run_id}.json"
        return dcc.send_string(content, filename=fname)
