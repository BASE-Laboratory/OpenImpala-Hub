"""Reactive Dash callbacks for The OpenImpala Hub (`v0.4.0` — BASE Lab Signature)."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
from typing import Any
import zipfile

from basecoat import write_ome_zarr
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
import numpy as np

from .. import __version__ as _HUB_VERSION
from ..contracts import MicrostructureSpec, SolveSpec
from ..engine.bundle import build_report_bundle_bytes
from ..engine.chunked_io import (
    dataset_intensity_range,
    dataset_structural_metrics,
    extract_native_roi_volume,
    fit_reference_circle,
    load_chunked_volume,
    load_echem_csv,
    load_lazy_slice,
    render_slice_with_roi_overlay,
    stack_dimensions,
)
from ..engine.electrochem import evaluate_cell_electrochemistry
from ..engine.mechanics import has_sms_backend
from ..engine.operando_demo import discover_operando_datasets
from ..engine.solver import ExecutedRun, run_transport_pipeline
from ..engine.steering import compute_cell_resistance, suggest_next_setpoint_gp
from ..engine.uploads import inspect_staged_volume, stage_upload_bytes
from ..presets import PRESETS, make_carbon_felt_spec
from .figures import (
    SlicePlane,
    build_3d_isosurface_figure,
    build_chunked_mesh3d_figure,
    build_compact_discharge_figure,
    build_directional_and_profile_figure,
    build_electrochem_figure,
    build_gp_steering_figure,
    build_morphology_figure,
    build_operando_echem_figure,
    build_operando_structural_figure,
    build_slice_histogram_figure,
    build_upload_segmentation_preview_figure,
    empty_placeholder_figure,
    render_run_slice_png,
)
from .layout import history_entry_from_summary
from .store import RunRegistry


def _history_entry_from_run(run: ExecutedRun) -> dict[str, Any]:
    return history_entry_from_summary(run.summary)


def _append_history(
    history: list[dict[str, Any]] | None,
    entry: dict[str, Any],
    max_len: int = 36,
) -> list[dict[str, Any]]:
    items = [h for h in (history or []) if h.get("run_id") != entry["run_id"]]
    items.append(entry)
    return items[-max_len:]


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

    if active_source_kind in ("user_upload", "benchmark_dataset"):
        v_pts = [h for h in items if h.get("source_kind") == active_source_kind]
        if len(v_pts) >= 2:
            xs = [float(p.get("virtual_compression_strain", 0.0)) for p in v_pts]
            if len(set(round(x, 3) for x in xs)) >= 2:
                ys = [
                    _recompute_entry_r_cell(
                        p,
                        w_ion=w_ion,
                        w_elec=w_elec,
                        w_kin=w_kin,
                        sigma_over_kappa_bulk=sigma_over_kappa_bulk,
                    )
                    for p in v_pts
                ]
                return suggest_next_setpoint_gp(
                    xs,
                    ys,
                    bounds=(0.0, 0.50),
                    parameter_name="Virtual Z-strain",
                    parameter_unit="-",
                )

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


def serialize_campaign_json(history: list[dict[str, Any]]) -> str:
    """Serialize `history` into a versioned portable campaign JSON document."""
    doc = {
        "schema": "openimpala_hub.campaign.v1",
        "hub_version": _HUB_VERSION,
        "exported_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs": list(history or []),
    }
    return json.dumps(doc, indent=2)


def parse_campaign_json(raw_text: str) -> list[dict[str, Any]]:
    """Validate and extract campaign history entries from a JSON string."""
    parsed = json.loads(raw_text)
    if isinstance(parsed, dict) and isinstance(parsed.get("runs"), list):
        entries = parsed["runs"]
    elif isinstance(parsed, list):
        entries = parsed
    else:
        raise ValueError("Invalid campaign JSON: expected {'runs': [...]} or array.")

    valid: list[dict[str, Any]] = []
    for item in entries[:64]:
        if not isinstance(item, dict) or "run_id" not in item or "porosity" not in item:
            continue
        valid.append(dict(item))
    if not valid:
        raise ValueError("Campaign JSON contains no valid run entries.")
    return valid


def evaluate_teaching_lab_progress(history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Evaluate completion status of the 4 interactive image-based modelling exercises."""
    items = list(history or [])
    has_cube = any(
        h.get("source_kind") == "open_cube"
        and abs(float(h.get("porosity", 0.0)) - 1.0) < 1e-3
        and h.get("tau_z") is not None
        and abs(float(h["tau_z"]) - 1.0) < 1e-2
        for h in items
    )
    has_bruggeman = any(
        h.get("source_kind") in ("benchmark_dataset", "overlapping_spheres", "gaussian_blobs")
        and h.get("tau_z") is not None
        and float(h["tau_z"]) > max(float(h.get("porosity", 1.0)), 0.05) ** (-0.5)
        for h in items
    )
    felt_loads = {
        round(float(h.get("clamping_kpa", 0.0)), 1)
        for h in items
        if h.get("source_kind") in ("carbon_felt", "porousdiff_gallery")
    }
    has_sweep = len(felt_loads) >= 4
    has_high_rate_diff = any(
        h.get("limiting_c_rate") is not None and float(h["limiting_c_rate"]) < 4.0
        for h in items
    )

    return [
        {
            "step": "Exercise 1",
            "title": "Straight-channel sanity check",
            "passed": has_cube,
            "description": "Run an unobstructed open cube (100% pore space) and confirm pathway tortuosity equals 1.000 along every axis.",
        },
        {
            "step": "Exercise 2",
            "title": "Detours around spherical particles",
            "passed": has_bruggeman,
            "description": "Simulate a 3D sphere pack or foam and see how real pore bottlenecks exceed the textbook Bruggeman approximation.",
        },
        {
            "step": "Exercise 3",
            "title": "Compression trade-off sweep",
            "passed": has_sweep,
            "description": "Evaluate at least 4 compression loads on carbon felt to find the sweet spot between solid contact and open pores.",
        },
        {
            "step": "Exercise 4",
            "title": "Fast-charge salt depletion",
            "passed": has_high_rate_diff,
            "description": "Inspect a dense or compressed electrode where high tortuosity limits fast charging below 4C.",
        },
    ]


def _result_row(name: str, hint: str, val: str) -> html.Tr:
    return html.Tr(
        [
            html.Td(
                [
                    html.Span(name, className="metric-name"),
                    html.Span(hint, className="metric-hint"),
                ]
            ),
            html.Td(val, className="metric-val"),
        ]
    )


def _build_plain_results_rows(run: ExecutedRun) -> list[html.Tr]:
    s = run.summary
    tz = s.directional.get("z")
    morph = s.morphology or {}
    mech = s.mechanics or {}
    echem = s.electrochem or {}

    eps_str = f"{s.porosity * 100:.1f}% ({s.porosity:.4f})"
    perc_pct = float(morph.get("percolating_fraction", s.porosity)) * 100.0
    trap_pct = float(morph.get("trapped_fraction", 0.0)) * 100.0
    conn_str = f"{perc_pct:.1f}% open · {trap_pct:.1f}% trapped"

    tau_str = f"{tz.tortuosity:.3f}" if (tz and tz.tortuosity is not None) else "Blocked"
    deff_val = tz.effective_diffusivity if tz else 0.0
    deff_str = f"{deff_val * 100:.1f}% ({deff_val:.4f})"
    if s.solid_effective_conductivity_z > 0:
        sig_str = f"{s.solid_effective_conductivity_z:.4f}"
    elif s.porosity >= 0.999:
        sig_str = "N/A (all pore)"
    else:
        sig_str = "0.0000 (no Z bridge)"

    e_gpa = mech.get("effective_modulus_gpa")
    if e_gpa is not None:
        stiff_str = f"{float(e_gpa):.2f} GPa"
    else:
        stiff_str = "Off (enable SMS)" if has_sms_backend() else "Optional .[sms]"

    c_lim = echem.get("limiting_c_rate")
    clim_str = f"{float(c_lim):.2f}C" if c_lim is not None else "N/A"

    return [
        _result_row(
            "Open pore space (ε)",
            "Fraction of volume open to liquid electrolyte",
            eps_str,
        ),
        _result_row(
            "Connected vs trapped pores",
            "Trapped pores cannot carry steady-state current",
            conn_str,
        ),
        _result_row(
            "Pathway tortuosity (τ)",
            "Detour factor for ions through thickness (1.0 = straight)",
            tau_str,
        ),
        _result_row(
            "Effective ion diffusivity",
            "Ion transport speed relative to open liquid",
            deff_str,
        ),
        _result_row(
            "Solid electrical wiring",
            "Relative electron conduction through solid network",
            sig_str,
        ),
        _result_row(
            "Mechanical stiffness",
            "Through-plane Young's modulus under load",
            stiff_str,
        ),
        _result_row(
            "Max fast-charge rate",
            "Rate where electrolyte salt begins to deplete",
            clim_str,
        ),
    ]


def register_callbacks(app: Dash, registry: RunRegistry) -> None:
    """Attach all v0.4.0 callbacks onto `app`."""

    # ── 1. Clientside Theme Toggle (`BASE Lab Signature` <-> `Warm Darkroom`) ──
    app.clientside_callback(
        """
        function(nClicks, storedTheme) {
            const trig = (window.dash_clientside && window.dash_clientside.callback_context && window.dash_clientside.callback_context.triggered_id) || null;
            let theme = (storedTheme === "dark") ? "dark" : "light";
            if (trig === "theme-toggle-btn" && nClicks > 0) {
                theme = (theme === "dark") ? "light" : "dark";
            }
            document.documentElement.setAttribute("data-theme", theme);
            const btnLabel = (theme === "dark") ? "Light mode" : "Dark mode";
            return [theme, btnLabel, theme];
        }
        """,
        Output("theme-store", "data"),
        Output("theme-toggle-btn", "children"),
        Output("hub-theme-wrapper", "data-theme"),
        Input("theme-toggle-btn", "n_clicks"),
        State("theme-store", "data"),
    )

    # ── 2. Clientside 3D Opacity (`Plotly.restyle`, 0-roundtrip 60 fps) ──
    app.clientside_callback(
        """
        function(opacityVal) {
            const container = document.getElementById("volume-3d-graph");
            if (container && window.Plotly) {
                const plotEl = container.querySelector(".js-plotly-plot") || container;
                if (plotEl && plotEl.data && plotEl.data.length > 0) {
                    window.Plotly.restyle(plotEl, {opacity: opacityVal});
                }
            }
            return "Opacity " + Number(opacityVal).toFixed(2);
        }
        """,
        Output("vol-3d-caption-right", "title"),
        Input("vol-opacity-slider", "value"),
    )

    # ── 3. Top Navigation & Sample Source Panel Visibility ──
    @app.callback(
        Output("panel-mode-generate", "style"),
        Output("panel-mode-xray", "style"),
        Output("panel-mode-upload", "style"),
        Output("panel-3d-controls", "style"),
        Output("view-slices", "style"),
        Output("view-3d", "style"),
        Output("view-compare", "style"),
        Output("view-pores", "style"),
        Output("view-plan", "style"),
        Output("view-guide", "style"),
        Output("hub-stage-grid", "className"),
        Output("hub-right-column", "style"),
        Input("sample-mode-radio", "value"),
        Input("main-view-tabs", "value"),
    )
    def _sync_stage_panels(sample_mode: str, tab_val: str):
        show = {"display": "block"}
        hide = {"display": "none"}
        show_flex = {"display": "flex"}

        p_gen = show if sample_mode == "generate" else hide
        p_xray = show if sample_mode == "xray" else hide
        p_up = show if sample_mode == "upload" else hide
        p_3d = {"display": "block", "marginTop": "16px"} if tab_val == "tab-3d" else hide

        v_slices = show_flex if tab_val == "tab-slices" else hide
        v_3d = show_flex if tab_val == "tab-3d" else hide
        v_cmp = show if tab_val == "tab-compare" else hide
        v_pores = show if tab_val == "tab-pores" else hide
        v_plan = show if tab_val == "tab-plan" else hide
        v_guide = show if tab_val == "tab-guide" else hide

        is_wide = tab_val in ("tab-compare", "tab-pores", "tab-plan", "tab-guide")
        stage_cls = "hub-stage hub-stage--wide" if is_wide else "hub-stage"
        right_style = hide if is_wide else show
        return (
            p_gen,
            p_xray,
            p_up,
            p_3d,
            v_slices,
            v_3d,
            v_cmp,
            v_pores,
            v_plan,
            v_guide,
            stage_cls,
            right_style,
        )

    # ── 4. Open Selected X-ray Region Directly in 3D View ──
    @app.callback(
        Output("main-view-tabs", "value"),
        Output("vol-source-dropdown", "value"),
        Input("open-roi-3d-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _open_roi_in_3d(n_clicks: int):
        if not n_clicks:
            raise PreventUpdate
        return "tab-3d", "xray_roi"

    # ── 5. New Random Seed & Control Readouts ──
    @app.callback(
        Output("seed-store", "data"),
        Input("new-seed-btn", "n_clicks"),
        State("seed-store", "data"),
        prevent_initial_call=True,
    )
    def _bump_seed(n_clicks: int, cur_seed: int | None):
        if not n_clicks:
            raise PreventUpdate
        nxt = ((int(cur_seed or 42) * 73 + 19) % 8999) + 100
        return nxt

    @app.callback(
        Output("clamping-kpa-readout", "children"),
        Output("porosity-readout", "children"),
        Output("virtual-compress-readout", "children"),
        Output("vol-threshold-readout", "children"),
        Output("clamping-field-wrap", "style"),
        Output("porosity-field-wrap", "style"),
        Output("electrode-role-dropdown", "value"),
        Input("clamping-kpa-slider", "value"),
        Input("porosity-slider", "value"),
        Input("virtual-compress-slider", "value"),
        Input("vol-threshold-slider", "value"),
        Input("source-kind-dropdown", "value"),
        State("electrode-role-dropdown", "value"),
    )
    def _sync_control_readouts(
        kpa: float,
        eps: float,
        vcomp: float,
        thr: float,
        source_kind: str,
        cur_role: str,
    ):
        show_kpa = source_kind in ("porousdiff_gallery", "carbon_felt")
        show_eps = source_kind not in ("open_cube", "benchmark_dataset")
        role_out = cur_role
        if ctx.triggered_id == "source-kind-dropdown":
            if source_kind == "multiphase_electrode":
                role_out = "positive"
            elif source_kind in ("porousdiff_gallery", "carbon_felt"):
                role_out = "negative"
        return (
            f"{float(kpa or 0):.0f} kPa",
            f"{float(eps or 0.5) * 100:.0f}%",
            f"{float(vcomp or 0):.0f}%",
            f"{float(thr or 128):.0f}",
            {"display": "block"} if show_kpa else {"display": "none"},
            {"display": "block"} if show_eps else {"display": "none"},
            role_out,
        )

    # ── 6. Operando Dataset Bounds & ROI Readouts ──
    @app.callback(
        Output("operando-time-slider", "max"),
        Output("operando-time-slider", "marks"),
        Output("operando-time-slider", "value"),
        Output("operando-z-slider", "max"),
        Output("operando-z-slider", "marks"),
        Output("operando-z-slider", "value"),
        Output("roi-x-slider", "max"),
        Output("roi-x-slider", "marks"),
        Output("roi-x-slider", "value"),
        Output("roi-y-slider", "max"),
        Output("roi-y-slider", "marks"),
        Output("roi-y-slider", "value"),
        Output("roi-z-slider", "max"),
        Output("roi-z-slider", "marks"),
        Output("roi-z-slider", "value"),
        Input("operando-dataset-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _sync_operando_dataset_bounds(ds_key: str):
        datasets = discover_operando_datasets()
        ds = datasets.get(ds_key) or next(iter(datasets.values()))
        nz, ny, nx = stack_dimensions(str(ds.timepoints[0].path))
        tp_max = max(0, len(ds.timepoints) - 1)
        tp_marks = {
            i: tp.label.split("(")[0].strip()[:12]
            for i, tp in enumerate(ds.timepoints)
        }
        tp_val = min(1, tp_max)
        z_max = max(0, nz - 1)
        z_val = nz // 2
        z_marks = {0: "0", z_max // 2: str(z_max // 2), z_max: str(z_max)}

        if ds_key == "dls_i13_zn13x":
            rx = [min(nx - 20, 190), min(nx, 300)]
            ry = [min(ny - 20, 130), min(ny, 240)]
        else:
            half_xy = min(48, max(16, nx // 4))
            rx = [max(0, nx // 2 - half_xy), min(nx, nx // 2 + half_xy)]
            ry = [max(0, ny // 2 - half_xy), min(ny, ny // 2 + half_xy)]
        half_z = min(20, max(6, nz // 4))
        rz = [max(0, z_val - half_z), min(nz, z_val + half_z)]
        return (
            tp_max,
            tp_marks,
            tp_val,
            z_max,
            z_marks,
            z_val,
            nx,
            {0: "0", nx: str(nx)},
            rx,
            ny,
            {0: "0", ny: str(ny)},
            ry,
            nz,
            {0: "0", nz: str(nz)},
            rz,
        )

    @app.callback(
        Output("operando-time-readout", "children"),
        Output("operando-z-readout", "children"),
        Output("roi-dims-readout", "children"),
        Input("operando-dataset-dropdown", "value"),
        Input("operando-time-slider", "value"),
        Input("operando-z-slider", "value"),
        Input("roi-x-slider", "value"),
        Input("roi-y-slider", "value"),
        Input("roi-z-slider", "value"),
    )
    def _sync_operando_readouts(
        ds_key: str,
        tp_idx: int,
        z_idx: int,
        rx: list[int],
        ry: list[int],
        rz: list[int],
    ):
        datasets = discover_operando_datasets()
        ds = datasets.get(ds_key) or next(iter(datasets.values()))
        idx = int(np.clip(tp_idx or 0, 0, len(ds.timepoints) - 1))
        tp = ds.timepoints[idx]
        nz, _, _ = stack_dimensions(str(tp.path))
        dx = max(1, int(rx[1]) - int(rx[0]))
        dy = max(1, int(ry[1]) - int(ry[0]))
        dz = max(1, int(rz[1]) - int(rz[0]))
        v_um = ds.voxel_size_um
        roi_str = (
            f"Box: {dx} x {dy} x {dz} px "
            f"({dx * v_um:.0f} x {dy * v_um:.0f} x {dz * v_um:.0f} µm at 1x1x1)"
        )
        return tp.label, f"{int(z_idx or 0)} of {max(0, nz - 1)}", roi_str

    # ── 7. Upload Staging & Threshold Preview ──
    @app.callback(
        Output("uploaded-file-store", "data"),
        Output("upload-status-text", "children"),
        Output("upload-dz-um", "value"),
        Output("upload-dy-um", "value"),
        Output("upload-dx-um", "value"),
        Output("upload-manual-thr", "min"),
        Output("upload-manual-thr", "max"),
        Output("upload-manual-thr", "step"),
        Output("upload-manual-thr", "value"),
        Input("upload-volume-box", "contents"),
        State("upload-volume-box", "filename"),
        prevent_initial_call=True,
    )
    def _stage_user_upload(contents: str | None, filename: str | None):
        if not contents:
            raise PreventUpdate
        fname = filename or "uploaded_volume.tif"
        try:
            _, b64 = contents.split(",", 1) if "," in contents else ("", contents)
            raw_bytes = base64.b64decode(b64)
            staged_path = stage_upload_bytes(raw_bytes, fname)
            info = inspect_staged_volume(staged_path, display_name=fname)
            cal = info.get("calibrated_voxel_um_zyx") or [1.0, 1.0, 1.0]
            vmin = float(info.get("val_min", 0.0))
            vmax = float(info.get("val_max", 255.0))
            if vmax <= vmin:
                vmax = vmin + 1.0
            otsu = float(info.get("otsu_threshold", 0.5 * (vmin + vmax)))
            step = round(max((vmax - vmin) / 100.0, 0.01), 4)
            shp = info["shape_zyx"]
            msg = f"Ready: {fname} ({shp[2]} x {shp[1]} x {shp[0]} px, {info['dtype']})"
            return (
                {"upload_path": str(staged_path), "filename": fname, "inspection": info},
                msg,
                float(cal[0]),
                float(cal[1]),
                float(cal[2]),
                vmin,
                vmax,
                step,
                otsu,
            )
        except Exception as exc:
            return None, f"Upload error: {exc}", 1.0, 1.0, 1.0, 0.0, 255.0, 1.0, 128.0

    @app.callback(
        Output("upload-preview-graph", "figure"),
        Output("upload-manual-thr-wrap", "style"),
        Input("uploaded-file-store", "data"),
        Input("upload-seg-mode", "value"),
        Input("upload-manual-thr", "value"),
        Input("upload-contrast-radio", "value"),
        Input("theme-store", "data"),
    )
    def _render_upload_preview(
        up_data: dict | None,
        seg_mode: str,
        manual_thr: float,
        contrast: str,
        theme: str,
    ):
        dark = theme == "dark"
        thr_style = {"display": "block"} if seg_mode == "manual" else {"display": "none"}
        if not up_data or not up_data.get("inspection"):
            return (
                empty_placeholder_figure(
                    "Drop a 3D file above to preview threshold",
                    height=175,
                    dark_mode=dark,
                ),
                thr_style,
            )
        fig = build_upload_segmentation_preview_figure(
            up_data["inspection"],
            seg_mode=seg_mode,
            manual_threshold=float(manual_thr or 0.0),
            dark_is_pore=(contrast != "bright"),
            height=175,
            dark_mode=dark,
        )
        return fig, thr_style

    # ── 8. Unified Simulation Executor & Slice-Mode Switcher ──
    @app.callback(
        Output("active-run-id-store", "data"),
        Output("run-history-store", "data"),
        Output("slice-display-mode", "value"),
        Input("sample-mode-radio", "value"),
        Input("run-generate-btn", "n_clicks"),
        Input("new-seed-btn", "n_clicks"),
        Input("simulate-roi-btn", "n_clicks"),
        Input("run-upload-btn", "n_clicks"),
        Input("run-suggested-load-btn", "n_clicks"),
        Input("run-5pt-sweep-btn", "n_clicks"),
        Input("guide-sweep-btn", "n_clicks"),
        Input("preset-btn-cube", "n_clicks"),
        Input("preset-btn-spheres", "n_clicks"),
        Input("preset-btn-cathode", "n_clicks"),
        Input("load-campaign-upload", "contents"),
        State("source-kind-dropdown", "value"),
        State("clamping-kpa-slider", "value"),
        State("porosity-slider", "value"),
        State("grid-size-dropdown", "value"),
        State("voxel-um-input", "value"),
        State("electrode-role-dropdown", "value"),
        State("physics-options-check", "value"),
        State("chemistry-dropdown", "value"),
        State("seed-store", "data"),
        State("operando-dataset-dropdown", "value"),
        State("operando-time-slider", "value"),
        State("roi-x-slider", "value"),
        State("roi-y-slider", "value"),
        State("roi-z-slider", "value"),
        State("uploaded-file-store", "data"),
        State("upload-seg-mode", "value"),
        State("upload-manual-thr", "value"),
        State("upload-contrast-radio", "value"),
        State("upload-dz-um", "value"),
        State("upload-dy-um", "value"),
        State("upload-dx-um", "value"),
        State("virtual-compress-slider", "value"),
        State("suggested-kpa-store", "data"),
        State("active-run-id-store", "data"),
        State("run-history-store", "data"),
        prevent_initial_call=True,
    )
    def _execute_simulation_or_mode_switch(
        sample_mode: str,
        _n_gen: int,
        _n_seed: int,
        _n_roi: int,
        _n_up: int,
        _n_sug: int,
        _n_swp1: int,
        _n_swp2: int,
        _n_cube: int,
        _n_sph: int,
        _n_cath: int,
        campaign_contents: str | None,
        source_kind: str,
        clamping_kpa: float,
        porosity: float,
        grid_side: int,
        voxel_um: float,
        electrode_role: str,
        phys_opts: list[str] | None,
        chem_key: str,
        seed_val: int,
        ds_key: str,
        tp_idx: int,
        rx: list[int],
        ry: list[int],
        rz: list[int],
        up_data: dict | None,
        up_seg_mode: str,
        up_manual_thr: float,
        up_contrast: str,
        up_dz: float,
        up_dy: float,
        up_dx: float,
        vcomp_pct: float,
        suggested_kpa: float,
        cur_run_id: str | None,
        history: list[dict] | None,
    ):
        trig = ctx.triggered_id
        if trig == "sample-mode-radio":
            return (
                no_update,
                no_update,
                "xray" if sample_mode == "xray" else "sim",
            )

        opts = set(phys_opts or [])
        want_sms = "sms" in opts and has_sms_backend()
        want_homog = "homog" in opts
        want_rev = "rev" in opts
        side = max(16, min(96, int(grid_side or 48)))
        v_um = max(0.05, float(voxel_um or 1.6))
        seed = int(seed_val or 42)
        if trig == "new-seed-btn":
            seed = ((seed * 73 + 19) % 8999) + 100

        # Campaign JSON import
        if trig == "load-campaign-upload":
            if not campaign_contents:
                raise PreventUpdate
            _, b64 = (
                campaign_contents.split(",", 1)
                if "," in campaign_contents
                else ("", campaign_contents)
            )
            raw_json = base64.b64decode(b64).decode("utf-8")
            imported = parse_campaign_json(raw_json)
            last_id = imported[-1]["run_id"]
            registry.get_or_rehydrate(last_id, imported)
            return last_id, imported, "sim"

        # 1-Click Guide Presets
        if trig in ("preset-btn-cube", "preset-btn-spheres", "preset-btn-cathode"):
            pkey = {
                "preset-btn-cube": "teaching_analytical_cube",
                "preset-btn-spheres": "teaching_sphere_pack",
                "preset-btn-cathode": "composite_cathode_3phase",
            }[trig]
            preset = PRESETS[pkey]
            p_side = min(48, preset.microstructure.shape[0])
            ms = MicrostructureSpec(
                **{**preset.microstructure.to_dict(), "shape": (p_side, p_side, p_side)}
            )
            ss = SolveSpec(
                **{
                    **preset.solve.to_dict(),
                    "compute_mechanics": want_sms and ms.source_kind != "open_cube",
                    "chemistry_preset": chem_key or "nmc811_li_ion",
                }
            )
            executed = run_transport_pipeline(ms, ss)
            rid = registry.put(executed)
            return rid, _append_history(history, _history_entry_from_run(executed)), "sim"

        # 5-Load Compression Sweep
        if trig in ("run-5pt-sweep-btn", "guide-sweep-btn"):
            hist_out = list(history or [])
            last_id = cur_run_id
            for load_kpa in (35.0, 95.0, 165.0, 260.0, 410.0):
                ms = make_carbon_felt_spec(load_kpa, shape=(36, 36, 36), seed=seed)
                ss = SolveSpec(
                    directions=("x", "y", "z"),
                    solid_phases=(1, 2),
                    compute_mechanics=want_sms,
                    chemistry_preset=chem_key or "vrfb_carbon_felt",
                )
                ex = run_transport_pipeline(ms, ss)
                last_id = registry.put(ex)
                hist_out = _append_history(hist_out, _history_entry_from_run(ex))
            return last_id, hist_out, "sim"

        # Bayesian GP Recommended Load
        if trig == "run-suggested-load-btn":
            kpa_run = float(suggested_kpa or 220.0)
            ms = MicrostructureSpec(
                source_kind="porousdiff_gallery",
                label=f"PorousDiff Carbon Felt ({kpa_run:.0f} kPa)",
                shape=(side, side, side),
                voxel_size_um=(v_um, v_um, v_um),
                porosity=float(porosity or 0.78),
                clamping_kpa=kpa_run,
                compression_strain=-1.0,
                seed=seed,
                electrode_role=electrode_role or "negative",
            )
            ss = SolveSpec(
                directions=("x", "y", "z"),
                solid_phases=(1, 2),
                compute_mechanics=want_sms,
                chemistry_preset=chem_key or "nmc811_li_ion",
            )
            ex = run_transport_pipeline(ms, ss)
            rid = registry.put(ex)
            return rid, _append_history(history, _history_entry_from_run(ex)), "sim"

        # Simulate Selected Native 1x1x1 X-ray Region (BatteryExplorer -> OpenImpala + SMS)
        if trig == "simulate-roi-btn":
            datasets = discover_operando_datasets()
            ds = datasets.get(ds_key) or next(iter(datasets.values()))
            idx = int(np.clip(tp_idx or 0, 0, len(ds.timepoints) - 1))
            tp = ds.timepoints[idx]
            pristine_path = str(ds.timepoints[0].path)
            roi_vol, thr_used = extract_native_roi_volume(
                str(tp.path),
                x_range=(int(rx[0]), int(rx[1])),
                y_range=(int(ry[0]), int(ry[1])),
                z_range=(int(rz[0]), int(rz[1])),
                voxel_size_um=ds.voxel_size_um,
                threshold_mode="midpoint",
                pristine_ref_path=pristine_path,
            )
            ms = MicrostructureSpec(
                source_kind="benchmark_dataset",
                label=f"X-ray region ({tp.label}, Z {rz[0]}–{rz[1]})",
                shape=roi_vol.shape,
                voxel_size_um=(ds.voxel_size_um, ds.voxel_size_um, ds.voxel_size_um),
                electrode_role="negative",
            )
            ss = SolveSpec(
                conducting_phase=0,
                solid_phases=(1,),
                directions=("x", "y", "z"),
                compute_homogenisation=want_homog,
                compute_rev_study=want_rev,
                compute_mechanics=want_sms,
                chemistry_preset=chem_key or "nmc811_li_ion",
            )
            ex = run_transport_pipeline(ms, ss, existing_volume=roi_vol)
            rid = registry.put(ex)
            return rid, _append_history(history, _history_entry_from_run(ex)), "sim"

        # Simulate Uploaded Experimental Scan
        if trig == "run-upload-btn":
            if not up_data or not up_data.get("upload_path"):
                raise PreventUpdate
            thr_spec: float | str | None
            if up_seg_mode == "manual":
                thr_spec = float(up_manual_thr or 128.0)
            elif up_seg_mode == "labels":
                thr_spec = "labels"
            else:
                thr_spec = "otsu"

            ms = MicrostructureSpec(
                source_kind="user_upload",
                label=f"Uploaded scan ({up_data.get('filename', 'volume')})",
                upload_path=str(up_data["upload_path"]),
                voxel_size_um=(
                    max(0.01, float(up_dz or 1.0)),
                    max(0.01, float(up_dy or 1.0)),
                    max(0.01, float(up_dx or 1.0)),
                ),
                threshold=thr_spec,
                dark_is_pore=(up_contrast != "bright"),
                virtual_compression_strain=round(float(vcomp_pct or 0.0) / 100.0, 4),
                electrode_role=electrode_role or "negative",
            )
            ss = SolveSpec(
                conducting_phase=0,
                solid_phases=(1, 2),
                directions=("x", "y", "z"),
                compute_homogenisation=want_homog,
                compute_rev_study=want_rev,
                compute_mechanics=want_sms,
                chemistry_preset=chem_key or "nmc811_li_ion",
            )
            ex = run_transport_pipeline(ms, ss)
            rid = registry.put(ex)
            return rid, _append_history(history, _history_entry_from_run(ex)), "sim"

        kpa_val = float(clamping_kpa if clamping_kpa is not None else 150.0)
        eps_val = float(porosity if porosity is not None else 0.78)
        label_map = {
            "porousdiff_gallery": f"PorousDiff Carbon Felt ({kpa_val:.0f} kPa)",
            "two_phase_xct": "Real 2-Phase Porous XCT (100³)",
            "carbon_felt": f"Custom Carbon Felt ({kpa_val:.0f} kPa)",
            "multiphase_electrode": f"3-Phase Cathode ({eps_val * 100:.0f}% pore)",
            "overlapping_spheres": f"Sphere Packing ({eps_val * 100:.0f}% pore)",
            "gaussian_blobs": f"Porous Foam ({eps_val * 100:.0f}% pore)",
            "benchmark_dataset": "Benchmark 8-Sphere Pack",
            "open_cube": "Straight-Channel Test Cube",
        }
        actual_kind = "benchmark_dataset" if source_kind == "two_phase_xct" else source_kind
        ds_name_arg = (
            "two_phase_stack"
            if source_kind == "two_phase_xct"
            else ("sphere_pack" if source_kind == "benchmark_dataset" else None)
        )
        solid_phases = () if actual_kind == "open_cube" else (1, 2)
        phase_diffs = (
            {0: 1.0, 1: 0.0, 2: 0.18}
            if actual_kind == "multiphase_electrode"
            else {0: 1.0, 1: 0.0}
        )
        ms = MicrostructureSpec(
            source_kind=actual_kind,  # type: ignore[arg-type]
            dataset_name=ds_name_arg,
            label=label_map.get(source_kind, source_kind),
            shape=(side, side, side),
            voxel_size_um=(v_um, v_um, v_um),
            seed=seed,
            porosity=1.0 if actual_kind == "open_cube" else eps_val,
            clamping_kpa=kpa_val,
            compression_strain=-1.0 if actual_kind in ("carbon_felt", "porousdiff_gallery") else 0.0,
            electrode_role=electrode_role or "negative",
        )
        ss = SolveSpec(
            conducting_phase=0,
            solid_phases=solid_phases,
            phase_diffusivities=phase_diffs,
            directions=("x", "y", "z"),
            compute_homogenisation=want_homog or source_kind == "multiphase_electrode",
            compute_rev_study=want_rev,
            compute_solid_wiring=(source_kind != "open_cube"),
            compute_mechanics=want_sms and source_kind != "open_cube",
            chemistry_preset=chem_key or "nmc811_li_ion",
        )
        ex = run_transport_pipeline(ms, ss)
        rid = registry.put(ex)
        return rid, _append_history(history, _history_entry_from_run(ex)), "sim"

    # ── 9. Solved-Sample Slice Slider Bounds & Jump to Bottleneck ──
    @app.callback(
        Output("run-slice-slider", "max"),
        Output("run-slice-slider", "value"),
        Output("run-slice-slider", "marks"),
        Input("active-run-id-store", "data"),
        Input("slice-plane-radio", "value"),
        Input("jump-roi-btn", "n_clicks"),
        State("run-history-store", "data"),
    )
    def _sync_run_slice_slider(
        run_id: str | None,
        plane: str,
        _n_jump: int,
        history: list[dict] | None,
    ):
        run = registry.get_or_rehydrate(run_id, history)
        if run is None:
            return 47, 24, {0: "0", 24: "24", 47: "47"}
        max_idx, mid_idx = plane_slider_bounds_for_run(run, plane)
        marks = {0: "0", max_idx // 2: str(max_idx // 2), max_idx: str(max_idx)}
        return max_idx, mid_idx, marks

    # ── 10. Primary Viewport Renderer (Lazy X-ray PNG OR Solved Sample PNG) ──
    @app.callback(
        Output("primary-slice-img", "src"),
        Output("primary-caption-left", "children"),
        Output("primary-caption-right", "children"),
        Output("slice-plane-radio", "style"),
        Output("slice-layer-radio", "style"),
        Output("sim-slice-scrub-row", "style"),
        Input("slice-display-mode", "value"),
        Input("active-run-id-store", "data"),
        Input("slice-plane-radio", "value"),
        Input("slice-layer-radio", "value"),
        Input("run-slice-slider", "value"),
        Input("operando-dataset-dropdown", "value"),
        Input("operando-cmap-dropdown", "value"),
        Input("operando-time-slider", "value"),
        Input("operando-z-slider", "value"),
        Input("roi-x-slider", "value"),
        Input("roi-y-slider", "value"),
        Input("roi-z-slider", "value"),
        State("run-history-store", "data"),
    )
    def _render_primary_viewport(
        disp_mode: str,
        run_id: str | None,
        plane: str,
        layer: str,
        s_idx: int,
        ds_key: str,
        cmap: str,
        tp_idx: int,
        z_idx: int,
        rx: list[int],
        ry: list[int],
        rz: list[int],
        history: list[dict] | None,
    ):
        if disp_mode == "xray":
            datasets = discover_operando_datasets()
            ds = datasets.get(ds_key) or next(iter(datasets.values()))
            idx = int(np.clip(tp_idx or 0, 0, len(ds.timepoints) - 1))
            tp = ds.timepoints[idx]
            zc = int(z_idx or 0)
            s_dict = load_lazy_slice(str(tp.path), zc, cmap=cmap or "grayscale")
            in_z = int(rz[0]) <= zc <= int(rz[1])
            box = (int(rx[0]), int(rx[1]), int(ry[0]), int(ry[1])) if in_z else None
            src = render_slice_with_roi_overlay(s_dict, roi_box_xy=box)
            h_px, w_px = s_dict["orig_hw"]
            fov_um = f"{w_px * ds.voxel_size_um:.0f} x {h_px * ds.voxel_size_um:.0f} µm"
            cap_l = f"{ds.label} · {tp.label} · XY slice {zc}"
            cap_r = f"{fov_um} · {s_dict['png_bytes'] // 1024} KB PNG ({s_dict['read_ms']:.1f} ms)"
            return (
                src,
                cap_l,
                cap_r,
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
            )

        run = registry.get_or_rehydrate(run_id, history)
        if run is None:
            raise PreventUpdate
        png_src, cap = render_run_slice_png(
            run,
            plane=plane or "xz",  # type: ignore[arg-type]
            slice_index=int(s_idx or 0),
            layer=layer or "flux",  # type: ignore[arg-type]
        )
        nz, ny, nx = run.phase_volume.shape
        cap_r = f"{nx} x {ny} x {nz} px"
        return (
            png_src,
            cap,
            cap_r,
            {"display": "inline-flex"},
            {"display": "inline-flex"},
            {
                "width": "100%",
                "maxWidth": "680px",
                "marginTop": "12px",
                "display": "grid",
                "gridTemplateColumns": "1fr auto",
                "gap": "12px",
                "alignItems": "center",
            },
        )

    # ── 11. Right-Column Plain-English Results Table & Context Plots ──
    @app.callback(
        Output("active-sample-subtitle", "children"),
        Output("results-summary-table", "children"),
        Output("tech-telemetry-line", "children"),
        Output("pybamm-bpx-preview", "children"),
        Output("compact-discharge-graph", "figure"),
        Output("operando-echem-graph", "figure"),
        Output("operando-structural-graph", "figure"),
        Output("right-panel-sim-chart", "style"),
        Output("right-panel-xray-charts", "style"),
        Output("right-panel-3d-hist", "style"),
        Input("active-run-id-store", "data"),
        Input("slice-display-mode", "value"),
        Input("main-view-tabs", "value"),
        Input("operando-dataset-dropdown", "value"),
        Input("operando-time-slider", "value"),
        Input("theme-store", "data"),
        State("run-history-store", "data"),
    )
    def _render_right_column(
        run_id: str | None,
        disp_mode: str,
        tab_val: str,
        ds_key: str,
        tp_idx: int,
        theme: str,
        history: list[dict] | None,
    ):
        dark = theme == "dark"
        run = registry.get_or_rehydrate(run_id, history)
        if run is None:
            raise PreventUpdate

        s = run.summary
        subtitle = (
            f"Showing: {s.label} · "
            f"{s.shape_zyx[2]} x {s.shape_zyx[1]} x {s.shape_zyx[0]} px "
            f"at {s.voxel_size_um_zyx[0]:.2f} µm/px"
        )
        rows = html.Tbody(_build_plain_results_rows(run))
        tech_line = (
            f"Run {s.run_id} · origin: {s.data_origin} · role: {s.electrode_role} · "
            f"solver: {s.solver_backend} ({s.solve_time_s * 1000:.0f} ms)"
        )
        bpx_text = json.dumps(
            {
                "pybamm_parameters": s.pybamm_parameters,
                "bpx_excerpt": s.bpx_parameters.get("bpx", {}).get("Parameter", {}),
            },
            indent=2,
        )
        fig_dis = build_compact_discharge_figure(s.electrochem, dark_mode=dark)

        show = {"display": "block"}
        hide = {"display": "none"}

        if tab_val == "tab-3d":
            return (
                subtitle,
                rows,
                tech_line,
                bpx_text,
                fig_dis,
                no_update,
                no_update,
                hide,
                hide,
                show,
            )

        if disp_mode == "xray":
            datasets = discover_operando_datasets()
            ds = datasets.get(ds_key) or next(iter(datasets.values()))
            idx = int(np.clip(tp_idx or 0, 0, len(ds.timepoints) - 1))
            tp = ds.timepoints[idx]
            df = load_echem_csv(str(ds.echem_csv))
            paths = tuple(str(t.path) for t in ds.timepoints)
            t_hours = [t.t_h for t in ds.timepoints]
            m_struct = dataset_structural_metrics(paths)
            fig_ec = build_operando_echem_figure(df, tp.t_h, dark_mode=dark)
            fig_st = build_operando_structural_figure(m_struct, t_hours, tp.t_h, dark_mode=dark)
            return (
                subtitle,
                rows,
                tech_line,
                bpx_text,
                fig_dis,
                fig_ec,
                fig_st,
                hide,
                show,
                hide,
            )

        return (
            subtitle,
            rows,
            tech_line,
            bpx_text,
            fig_dis,
            empty_placeholder_figure("Switch to X-ray Scans to view cycling traces", height=180, dark_mode=dark),
            empty_placeholder_figure("", height=180, dark_mode=dark),
            show,
            hide,
            hide,
        )

    # ── 12. 3D View Callback (Active Simulated Sample OR Dask-Pooled X-ray Stack/ROI) ──
    @app.callback(
        Output("volume-3d-graph", "figure"),
        Output("vol-3d-hist-graph", "figure"),
        Output("vol-3d-caption-left", "children"),
        Output("vol-3d-caption-right", "children"),
        Input("main-view-tabs", "value"),
        Input("vol-source-dropdown", "value"),
        Input("vol-threshold-slider", "value"),
        Input("vol-rim-slider", "value"),
        Input("vol-clip-x", "value"),
        Input("vol-clip-y", "value"),
        Input("vol-clip-z", "value"),
        Input("active-run-id-store", "data"),
        Input("operando-dataset-dropdown", "value"),
        Input("operando-time-slider", "value"),
        Input("roi-x-slider", "value"),
        Input("roi-y-slider", "value"),
        Input("roi-z-slider", "value"),
        Input("theme-store", "data"),
        State("vol-opacity-slider", "value"),
        State("run-history-store", "data"),
    )
    def _render_3d_view(
        tab_val: str,
        vol_src: str,
        thr: float,
        rim_cut: float,
        clip_x: list[float],
        clip_y: list[float],
        clip_z: list[float],
        run_id: str | None,
        ds_key: str,
        tp_idx: int,
        rx: list[int],
        ry: list[int],
        rz: list[int],
        theme: str,
        opacity: float,
        history: list[dict] | None,
    ):
        if tab_val != "tab-3d":
            raise PreventUpdate
        dark = theme == "dark"
        op_val = float(opacity if opacity is not None else 0.95)

        if vol_src in ("xray_full", "xray_roi"):
            datasets = discover_operando_datasets()
            ds = datasets.get(ds_key) or next(iter(datasets.values()))
            idx = int(np.clip(tp_idx or 0, 0, len(ds.timepoints) - 1))
            tp = ds.timepoints[idx]
            if vol_src == "xray_roi":
                v_dict = load_chunked_volume(
                    str(tp.path),
                    z_range=(int(rz[0]), int(rz[1])),
                    y_range=(int(ry[0]), int(ry[1])),
                    x_range=(int(rx[0]), int(rx[1])),
                    z_step=1,
                    xy_step=1,
                )
                circ = None
                mode_tag = "1x1x1 selected region"
            else:
                nz_tot, ny_tot, nx_tot = stack_dimensions(str(tp.path))
                z_st = 2 if nz_tot <= 96 else 4
                xy_st = 4 if nx_tot <= 256 else 16
                v_dict = load_chunked_volume(str(tp.path), z_step=z_st, xy_step=xy_st)
                circ = fit_reference_circle(str(ds.timepoints[0].path))
                mode_tag = f"pooled ({xy_st}x XY, {z_st}x Z)"

            xmin, xmax = float(v_dict["x"].min()), float(v_dict["x"].max())
            ymin, ymax = float(v_dict["y"].min()), float(v_dict["y"].max())
            zmin, zmax = float(v_dict["z"].min()), float(v_dict["z"].max())
            cbox = (
                (xmin + (xmax - xmin) * clip_x[0] / 100.0, xmin + (xmax - xmin) * clip_x[1] / 100.0),
                (ymin + (ymax - ymin) * clip_y[0] / 100.0, ymin + (ymax - ymin) * clip_y[1] / 100.0),
                (zmin + (zmax - zmin) * clip_z[0] / 100.0, zmin + (zmax - zmin) * clip_z[1] / 100.0),
            )
            fig3d = build_chunked_mesh3d_figure(
                v_dict,
                dark_mode=dark,
                threshold=float(thr or 128.0),
                opacity=op_val,
                rim_cutoff=float(rim_cut or 1.0),
                circle=circ,
                clip_box=cbox,
                height=540,
            )
            fighist = build_slice_histogram_figure(v_dict["hist"], v_dict["edges"], dark_mode=dark)
            shp = v_dict["shape"]
            return (
                fig3d,
                fighist,
                f"{ds.label} · {tp.label} ({mode_tag})",
                f"Mesh grid {shp[2]} x {shp[1]} x {shp[0]} ({v_dict['load_ms']:.0f} ms)",
            )

        run = registry.get_or_rehydrate(run_id, history)
        if run is None:
            raise PreventUpdate
        fig3d = build_3d_isosurface_figure(run, height=540, dark_mode=dark)
        if fig3d.data:
            fig3d.data[0].opacity = op_val
        arr = np.asarray(run.flux_field if run.flux_field is not None else run.phase_volume.data)
        hist, edges = np.histogram(arr[np.isfinite(arr)], bins=32)
        fighist = build_slice_histogram_figure(hist, edges, dark_mode=dark)
        nz, ny, nx = run.phase_volume.shape
        return (
            fig3d,
            fighist,
            f"{run.summary.label} · solid architecture & bottleneck box",
            f"{nx} x {ny} x {nz} px",
        )

    # ── 13. Compare View (Linked Z Sliders + X-ray Timepoints OR Simulated Runs) ──
    @app.callback(
        Output("cmp-left-select", "options"),
        Output("cmp-left-select", "value"),
        Output("cmp-right-select", "options"),
        Output("cmp-right-select", "value"),
        Output("cmp-left-z", "max"),
        Output("cmp-right-z", "max"),
        Input("cmp-mode-radio", "value"),
        Input("operando-dataset-dropdown", "value"),
        Input("run-history-store", "data"),
    )
    def _sync_compare_dropdowns(
        cmp_mode: str,
        ds_key: str,
        history: list[dict] | None,
    ):
        if cmp_mode == "xray":
            datasets = discover_operando_datasets()
            ds = datasets.get(ds_key) or next(iter(datasets.values()))
            opts = [{"label": tp.label, "value": str(i)} for i, tp in enumerate(ds.timepoints)]
            nz, _, _ = stack_dimensions(str(ds.timepoints[0].path))
            v_left = "0"
            v_right = "1" if len(opts) > 1 else "0"
            return opts, v_left, opts, v_right, max(0, nz - 1), max(0, nz - 1)

        items = list(history or [])
        opts = [
            {
                "label": f"{h.get('label', h['run_id'])} (ε={float(h['porosity']) * 100:.0f}%)",
                "value": str(h["run_id"]),
            }
            for h in items
        ]
        if not opts:
            return [], None, [], None, 47, 47
        v_left = opts[0]["value"]
        v_right = opts[-1]["value"]
        r_left = registry.get_or_rehydrate(v_left, items)
        r_right = registry.get_or_rehydrate(v_right, items)
        m_l = plane_slider_bounds_for_run(r_left, "xz")[0] if r_left else 47
        m_r = plane_slider_bounds_for_run(r_right, "xz")[0] if r_right else 47
        return opts, v_left, opts, v_right, m_l, m_r

    @app.callback(
        Output("cmp-left-z", "value"),
        Output("cmp-right-z", "value"),
        Input("cmp-left-z", "value"),
        Input("cmp-right-z", "value"),
        State("cmp-link-z-toggle", "value"),
        prevent_initial_call=True,
    )
    def _sync_linked_compare_z(zl: int, zr: int, link_vals: list[str] | None):
        if not link_vals or "link" not in link_vals:
            raise PreventUpdate
        if ctx.triggered_id == "cmp-left-z" and zl != zr:
            return zl, zl
        if ctx.triggered_id == "cmp-right-z" and zr != zl:
            return zr, zr
        raise PreventUpdate

    @app.callback(
        Output("cmp-left-img", "src"),
        Output("cmp-left-caption", "children"),
        Output("cmp-left-hist", "figure"),
        Output("cmp-right-img", "src"),
        Output("cmp-right-caption", "children"),
        Output("cmp-right-hist", "figure"),
        Output("cmp-delta-summary", "children"),
        Input("main-view-tabs", "value"),
        Input("cmp-mode-radio", "value"),
        Input("cmp-left-select", "value"),
        Input("cmp-right-select", "value"),
        Input("cmp-left-z", "value"),
        Input("cmp-right-z", "value"),
        Input("operando-dataset-dropdown", "value"),
        Input("operando-cmap-dropdown", "value"),
        Input("theme-store", "data"),
        State("run-history-store", "data"),
    )
    def _render_compare_view(
        tab_val: str,
        cmp_mode: str,
        left_sel: str | None,
        right_sel: str | None,
        zl: int,
        zr: int,
        ds_key: str,
        cmap: str,
        theme: str,
        history: list[dict] | None,
    ):
        if tab_val != "tab-compare":
            raise PreventUpdate
        dark = theme == "dark"

        if cmp_mode == "xray":
            datasets = discover_operando_datasets()
            ds = datasets.get(ds_key) or next(iter(datasets.values()))
            i_l = int(np.clip(int(left_sel or 0), 0, len(ds.timepoints) - 1))
            i_r = int(np.clip(int(right_sel or 0), 0, len(ds.timepoints) - 1))
            tp_l, tp_r = ds.timepoints[i_l], ds.timepoints[i_r]
            shared_rng = dataset_intensity_range(tuple(str(t.path) for t in ds.timepoints))
            sl = load_lazy_slice(str(tp_l.path), int(zl or 0), hist_range=shared_rng, cmap=cmap or "grayscale")
            sr = load_lazy_slice(str(tp_r.path), int(zr or 0), hist_range=shared_rng, cmap=cmap or "grayscale")
            hl = build_slice_histogram_figure(sl["hist"], sl["edges"], dark_mode=dark, x_range=shared_rng)
            hr = build_slice_histogram_figure(sr["hist"], sr["edges"], dark_mode=dark, x_range=shared_rng)
            m_struct = dataset_structural_metrics(tuple(str(t.path) for t in ds.timepoints))
            v_l = m_struct["low_atten_fraction"][i_l] * 100.0
            v_r = m_struct["low_atten_fraction"][i_r] * 100.0
            delta_txt = (
                f"Active-layer void + plated Li fraction: {tp_l.label} = {v_l:.2f}% vs "
                f"{tp_r.label} = {v_r:.2f}% (change: {v_r - v_l:+.2f} percentage points)."
            )
            return (
                sl["png_src"],
                f"{tp_l.label} · slice {int(zl or 0)} ({sl['read_ms']:.1f} ms)",
                hl,
                sr["png_src"],
                f"{tp_r.label} · slice {int(zr or 0)} ({sr['read_ms']:.1f} ms)",
                hr,
                html.Div(delta_txt, className="helper-copy"),
            )

        items = list(history or [])
        rl = registry.get_or_rehydrate(left_sel, items)
        rr = registry.get_or_rehydrate(right_sel, items)
        if rl is None or rr is None:
            raise PreventUpdate
        src_l, cap_l = render_run_slice_png(rl, plane="xz", slice_index=int(zl or 0), layer="flux")
        src_r, cap_r = render_run_slice_png(rr, plane="xz", slice_index=int(zr or 0), layer="flux")
        fl = np.asarray(rl.flux_field if rl.flux_field is not None else rl.phase_volume.data)
        fr = np.asarray(rr.flux_field if rr.flux_field is not None else rr.phase_volume.data)
        hl_arr, el_arr = np.histogram(fl[np.isfinite(fl)], bins=32)
        hr_arr, er_arr = np.histogram(fr[np.isfinite(fr)], bins=32)
        hl = build_slice_histogram_figure(hl_arr, el_arr, dark_mode=dark)
        hr = build_slice_histogram_figure(hr_arr, er_arr, dark_mode=dark)
        tl = rl.summary.directional.get("z")
        tr = rr.summary.directional.get("z")
        tau_l = tl.tortuosity if tl and tl.tortuosity else 0.0
        tau_r = tr.tortuosity if tr and tr.tortuosity else 0.0
        delta_table = html.Table(
            className="results-table",
            children=[
                html.Tbody(
                    [
                        _result_row(
                            "Open pore space comparison",
                            f"{rl.summary.label} vs {rr.summary.label}",
                            f"{rl.summary.porosity * 100:.1f}% -> {rr.summary.porosity * 100:.1f}%",
                        ),
                        _result_row(
                            "Pathway tortuosity comparison",
                            "Through-plane detour factor",
                            f"{tau_l:.3f} -> {tau_r:.3f}",
                        ),
                    ]
                )
            ],
        )
        return src_l, cap_l, hl, src_r, cap_r, hr, delta_table

    # ── 14. Discharge & Pores View ──
    @app.callback(
        Output("electrochem-graph", "figure"),
        Output("morphology-graph", "figure"),
        Output("directional-profile-graph", "figure"),
        Input("main-view-tabs", "value"),
        Input("active-run-id-store", "data"),
        Input("chemistry-dropdown", "value"),
        Input("theme-store", "data"),
        State("run-history-store", "data"),
    )
    def _render_pores_and_electrochem(
        tab_val: str,
        run_id: str | None,
        chem_key: str,
        theme: str,
        history: list[dict] | None,
    ):
        if tab_val != "tab-pores":
            raise PreventUpdate
        dark = theme == "dark"
        run = registry.get_or_rehydrate(run_id, history)
        if run is None:
            raise PreventUpdate
        s = run.summary
        tz = s.directional.get("z")
        d_eff_z = tz.effective_diffusivity if tz else s.porosity
        thick_m = float(s.shape_zyx[0]) * float(s.voxel_size_um_zyx[0]) * 1.0e-6
        ec_rep = evaluate_cell_electrochemistry(
            porosity=s.porosity,
            d_eff_z=d_eff_z,
            specific_surface_area_m_inv=max(s.interfacial_area_active_m_inv, 2.0e4),
            solid_sigma_eff_z=max(s.solid_effective_conductivity_z, 0.02),
            thickness_m=thick_m,
            chemistry_key=chem_key or "nmc811_li_ion",
        )
        fig_ec = build_electrochem_figure(ec_rep.to_dict(), dark_mode=dark)
        fig_mo = (
            build_morphology_figure(s.morphology, dark_mode=dark)
            if s.morphology
            else empty_placeholder_figure("No pore statistics", dark_mode=dark)
        )
        fig_dir = build_directional_and_profile_figure(run, dark_mode=dark)
        return fig_ec, fig_mo, fig_dir

    # ── 15. Plan Scans View (Bayesian GP Steering) ──
    @app.callback(
        Output("gp-steering-graph", "figure"),
        Output("gp-recommendation-banner", "children"),
        Output("suggested-kpa-store", "data"),
        Input("main-view-tabs", "value"),
        Input("run-history-store", "data"),
        Input("gp-w-ion", "value"),
        Input("gp-w-elec", "value"),
        Input("gp-w-kin", "value"),
        Input("theme-store", "data"),
    )
    def _render_gp_planner(
        tab_val: str,
        history: list[dict] | None,
        w_ion: float,
        w_elec: float,
        w_kin: float,
        theme: str,
    ):
        dark = theme == "dark"
        sug = compute_gp_for_campaign(
            history or [],
            active_source_kind="carbon_felt",
            w_ion=float(w_ion or 0.58),
            w_elec=float(w_elec or 0.27),
            w_kin=float(w_kin or 0.15),
        )
        if sug is None:
            return (
                empty_placeholder_figure(
                    "Run at least two compression loads to fit the optimiser",
                    dark_mode=dark,
                ),
                "Run at least two compression loads to get a suggested next test.",
                200.0,
            )
        fig = build_gp_steering_figure(sug, dark_mode=dark)
        banner = html.Div(
            [
                html.Strong(
                    f"Suggested next load: {sug.suggested_value:.0f} {sug.parameter_unit}"
                ),
                html.Span(
                    f" · Current best tested: {sug.current_best_parameter:.0f} {sug.parameter_unit} "
                    f"(resistance score {sug.current_best_r_cell:.3f}, {len(sug.observed_x)} loads tested)."
                ),
            ]
        )
        return fig, banner, float(sug.suggested_value)

    # ── 16. Guide View (Plain-English Interactive Exercises, Zero Emojis) ──
    @app.callback(
        Output("guide-cards-container", "children"),
        Input("run-history-store", "data"),
    )
    def _render_guide_cards(history: list[dict] | None):
        steps = evaluate_teaching_lab_progress(history)
        done_count = sum(1 for st in steps if st["passed"])
        cards = []
        for st in steps:
            tag_cls = "status-tag status-tag--done" if st["passed"] else "status-tag"
            tag_txt = "DONE" if st["passed"] else "TO TRY"
            cards.append(
                html.Div(
                    className="guide-card",
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "marginBottom": "6px",
                            },
                            children=[
                                html.Strong(f"{st['step']} · {st['title']}"),
                                html.Span(tag_txt, className=tag_cls),
                            ],
                        ),
                        html.Div(st["description"], className="helper-copy"),
                    ],
                )
            )
        header = html.Div(
            f"Completed {done_count} of {len(steps)} exercises in this session.",
            className="mono",
            style={"gridColumn": "1 / -1", "fontWeight": "600"},
        )
        return [header, *cards]

    # ── 17. Downloads: Report Bundle (.zip), BPX (.json), Campaign (.json), 3D Volume (.ome.zarr.zip) ──
    @app.callback(
        Output("download-bundle", "data"),
        Input("export-bundle-btn", "n_clicks"),
        State("active-run-id-store", "data"),
        State("run-history-store", "data"),
        prevent_initial_call=True,
    )
    def _download_bundle(n_clicks: int, run_id: str | None, history: list[dict] | None):
        if not n_clicks:
            raise PreventUpdate
        run = registry.get_or_rehydrate(run_id, history)
        if run is None:
            raise PreventUpdate
        raw = build_report_bundle_bytes(run, include_ome_zarr=True)
        return dcc.send_bytes(raw, f"openimpala_hub_{run.summary.run_id}.zip")

    @app.callback(
        Output("download-bpx", "data"),
        Input("export-bpx-btn", "n_clicks"),
        State("active-run-id-store", "data"),
        State("run-history-store", "data"),
        prevent_initial_call=True,
    )
    def _download_bpx(n_clicks: int, run_id: str | None, history: list[dict] | None):
        if not n_clicks:
            raise PreventUpdate
        run = registry.get_or_rehydrate(run_id, history)
        if run is None:
            raise PreventUpdate
        raw_str = json.dumps(run.summary.bpx_parameters, indent=2)
        return dict(
            content=raw_str,
            filename=f"bpx_parameters_{run.summary.run_id}.json",
            type="application/json",
        )

    @app.callback(
        Output("download-campaign", "data"),
        Input("save-campaign-btn", "n_clicks"),
        State("run-history-store", "data"),
        prevent_initial_call=True,
    )
    def _download_campaign(n_clicks: int, history: list[dict] | None):
        if not n_clicks:
            raise PreventUpdate
        return dict(
            content=serialize_campaign_json(history or []),
            filename="openimpala_campaign.json",
            type="application/json",
        )

    @app.callback(
        Output("download-volume", "data"),
        Input("save-volume-btn", "n_clicks"),
        State("active-run-id-store", "data"),
        State("run-history-store", "data"),
        prevent_initial_call=True,
    )
    def _download_volume_ome_zarr(n_clicks: int, run_id: str | None, history: list[dict] | None):
        if not n_clicks:
            raise PreventUpdate
        run = registry.get_or_rehydrate(run_id, history)
        if run is None:
            raise PreventUpdate
        with tempfile.TemporaryDirectory() as tmpd:
            zarr_dir = Path(tmpd) / f"sample_{run.summary.run_id}.ome.zarr"
            write_ome_zarr(run.phase_volume, zarr_dir, multiscale_levels=2, overwrite=True)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for fp in sorted(zarr_dir.rglob("*")):
                    if fp.is_file():
                        arc = f"{zarr_dir.name}/{fp.relative_to(zarr_dir).as_posix()}"
                        zf.write(fp, arcname=arc)
            return dcc.send_bytes(
                buf.getvalue(), f"sample_{run.summary.run_id}.ome.zarr.zip"
            )
