"""Plotly figure builders for 2D slices, flux overlays, 3D marching-cubes meshes, and GP steering."""

from __future__ import annotations

from typing import Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.measure import marching_cubes

from ..contracts import BayesianSetpointSuggestion
from ..engine.bridge import enforce_solve_grid_cap
from ..engine.solver import ExecutedRun

SlicePlane = Literal["xy", "xz", "yz"]
FieldLayer = Literal["phase", "potential", "flux"]


def _theme_tokens(dark_mode: bool) -> dict[str, str]:
    if dark_mode:
        return {
            "paper_bg": "#111827",
            "plot_bg": "#1f2937",
            "fg": "#f3f4f6",
            "muted": "#9ca3af",
            "grid": "#374151",
            "accent": "#38bdf8",
            "roi": "#2dd4bf",
        }
    return {
        "paper_bg": "#ffffff",
        "plot_bg": "#f8fafc",
        "fg": "#0f172a",
        "muted": "#475569",
        "grid": "#e2e8f0",
        "accent": "#2563eb",
        "roi": "#0d9488",
    }


def empty_placeholder_figure(message: str = "No run selected", *, height: int = 420, dark_mode: bool = False) -> go.Figure:
    tok = _theme_tokens(dark_mode)
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14, "color": tok["muted"]},
    )
    fig.update_layout(
        height=height,
        paper_bgcolor=tok["paper_bg"],
        plot_bgcolor=tok["plot_bg"],
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
    )
    return fig


def extract_2d_plane(
    arr_3d: np.ndarray,
    plane: SlicePlane,
    index: int,
) -> tuple[np.ndarray, int]:
    """Slice a `(Nz, Ny, Nx)` array along `plane`, clamping `index` safely."""
    nz, ny, nx = arr_3d.shape
    if plane == "xy":
        idx = min(max(0, int(index)), nz - 1)
        return arr_3d[idx, :, :], idx
    if plane == "xz":
        idx = min(max(0, int(index)), ny - 1)
        return arr_3d[:, idx, :], idx
    idx = min(max(0, int(index)), nx - 1)
    return arr_3d[:, :, idx], idx


def build_slice_figure(
    run: ExecutedRun,
    *,
    plane: SlicePlane = "xz",
    slice_index: int = 0,
    layer: FieldLayer = "flux",
    zmax_override: float | None = None,
    height: int = 460,
    dark_mode: bool = False,
) -> go.Figure:
    """Build an interactive 2D slice heatmap with calibrated micrometre axes and Zoom-ROI overlay."""
    tok = _theme_tokens(dark_mode)
    vol = run.phase_volume
    dz_um, dy_um, dx_um = vol.spacing_in("um")
    nz, ny, nx = vol.shape

    actual_layer = layer
    if actual_layer == "potential" and run.potential_field is None:
        actual_layer = "phase"
    if actual_layer == "flux" and run.flux_field is None:
        actual_layer = "phase"

    if actual_layer == "potential":
        src = run.potential_field
        colorscale = "RdBu_r"
        zmin, zmax = 0.0, 1.0
        cbar_title = "phi [-]"
        layer_title = "Steady-State Potential phi"
    elif actual_layer == "flux":
        src = run.flux_field
        colorscale = "Inferno"
        zmin = 0.0
        zmax = (
            float(zmax_override)
            if zmax_override is not None
            else max(float(np.percentile(src, 99.2)), 1.25)  # type: ignore[arg-type]
        )
        cbar_title = "|J / J0|"
        layer_title = "Diffusive Flux Crowding |J / J0|"
    else:
        src = vol.data
        max_phase = max(1, int(np.max(src)))
        colorscale = [
            [0.0, "#0f172a"],
            [0.5, "#64748b" if max_phase > 1 else "#38bdf8"],
            [1.0, "#f59e0b"],
        ]
        zmin, zmax = 0.0, float(max_phase)
        cbar_title = "Phase ID"
        layer_title = "Segmented Phase ID"

    slice_2d, clamped_idx = extract_2d_plane(src, plane, slice_index)  # type: ignore[arg-type]

    if plane == "xy":
        x_coords = (np.arange(nx) * dx_um).tolist()
        y_coords = (np.arange(ny) * dy_um).tolist()
        x_label, y_label = "x [µm]", "y [µm]"
        pos_um = clamped_idx * dz_um
        normal_desc = f"z = {clamped_idx} ({pos_um:.1f} µm)"
    elif plane == "xz":
        x_coords = (np.arange(nx) * dx_um).tolist()
        y_coords = (np.arange(nz) * dz_um).tolist()
        x_label, y_label = "x [µm]", "z [µm]"
        pos_um = clamped_idx * dy_um
        normal_desc = f"y = {clamped_idx} ({pos_um:.1f} µm)"
    else:
        x_coords = (np.arange(ny) * dy_um).tolist()
        y_coords = (np.arange(nz) * dz_um).tolist()
        x_label, y_label = "y [µm]", "z [µm]"
        pos_um = clamped_idx * dx_um
        normal_desc = f"x = {clamped_idx} ({pos_um:.1f} µm)"

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=slice_2d,
                x=x_coords,
                y=y_coords,
                zmin=zmin,
                zmax=zmax,
                colorscale=colorscale,
                colorbar={"title": cbar_title, "thickness": 14},
                hovertemplate=f"{x_label}: %{{x:.1f}}<br>{y_label}: %{{y:.1f}}<br>{cbar_title}: %{{z:.4g}}<extra></extra>",
            )
        ]
    )

    # Overlay Zoom-ROI #1 rectangle when the active slice intersects its normal bounds
    if run.summary.zoom_rois:
        roi = run.summary.zoom_rois[0]
        (z0, z1), (y0, y1), (x0, x1) = roi.bounds_zyx
        box: tuple[float, float, float, float] | None = None
        if plane == "xy" and z0 <= clamped_idx < z1:
            box = (x0 * dx_um, y0 * dy_um, (x1 - 1) * dx_um, (y1 - 1) * dy_um)
        elif plane == "xz" and y0 <= clamped_idx < y1:
            box = (x0 * dx_um, z0 * dz_um, (x1 - 1) * dx_um, (z1 - 1) * dz_um)
        elif plane == "yz" and x0 <= clamped_idx < x1:
            box = (y0 * dy_um, z0 * dz_um, (y1 - 1) * dy_um, (z1 - 1) * dz_um)

        if box is not None:
            fig.add_shape(
                type="rect",
                x0=box[0],
                y0=box[1],
                x1=box[2],
                y1=box[3],
                line={"color": "#00f5d4", "width": 2.4, "dash": "dash"},
            )
            fig.add_annotation(
                x=box[0],
                y=box[3],
                text="Zoom ROI #1",
                showarrow=False,
                yshift=10,
                xanchor="left",
                font={"size": 11, "color": "#00f5d4"},
                bgcolor="rgba(15, 23, 42, 0.78)",
            )

    fig.update_layout(
        title={"text": f"{layer_title} · {plane.upper()} plane ({normal_desc})", "font": {"size": 14}},
        height=height,
        paper_bgcolor=tok["paper_bg"],
        plot_bgcolor=tok["plot_bg"],
        font={"color": tok["fg"]},
        margin={"l": 56, "r": 36, "t": 46, "b": 48},
        xaxis={"title": x_label, "constrain": "domain", "gridcolor": tok["grid"]},
        yaxis={"title": y_label, "scaleanchor": "x", "scaleratio": 1.0, "gridcolor": tok["grid"]},
    )
    return fig


def build_directional_and_profile_figure(
    run: ExecutedRun,
    *,
    height: int = 460,
    dark_mode: bool = False,
) -> go.Figure:
    """Build a two-row summary chart: directional transport bars + through-thickness porosity profile."""
    tok = _theme_tokens(dark_mode)
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Directional Tortuosity & Effective Diffusivity",
            "Through-Thickness Slice Porosity Profile",
        ),
        vertical_spacing=0.18,
    )

    dirs = list(run.summary.directional.keys())
    taus = [
        run.summary.directional[d].tortuosity
        if run.summary.directional[d].tortuosity is not None
        else 0.0
        for d in dirs
    ]
    deffs = [run.summary.directional[d].effective_diffusivity for d in dirs]

    fig.add_trace(
        go.Bar(
            x=[d.upper() for d in dirs],
            y=taus,
            name="Tortuosity τ",
            marker_color="#3b82f6",
            text=[f"{v:.3f}" if v > 0 else "N/A" for v in taus],
            textposition="auto",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[d.upper() for d in dirs],
            y=deffs,
            name="D_eff / D_0",
            marker_color="#14b8a6",
            text=[f"{v:.3f}" for v in deffs],
            textposition="auto",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=run.summary.through_thickness_z_um,
            y=run.summary.through_thickness_porosity,
            mode="lines",
            name="Slice porosity ε(z)",
            line={"color": tok["accent"], "width": 2.2},
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=run.summary.porosity,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Bulk ε = {run.summary.porosity:.3f}",
        row=2,
        col=1,
    )

    fig.update_layout(
        height=height,
        barmode="group",
        paper_bgcolor=tok["paper_bg"],
        plot_bgcolor=tok["plot_bg"],
        font={"color": tok["fg"]},
        margin={"l": 52, "r": 24, "t": 44, "b": 44},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.06, "x": 0.0},
    )
    fig.update_xaxes(gridcolor=tok["grid"])
    fig.update_yaxes(gridcolor=tok["grid"])
    fig.update_xaxes(title_text="Coordinate along flow axis [µm]", row=2, col=1)
    fig.update_yaxes(title_text="Porosity [-]", row=2, col=1)
    return fig


def _wireframe_box_trace(
    bounds_um_zyx: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    *,
    name: str = "Zoom ROI #1",
    color: str = "#00f5d4",
) -> go.Scatter3d:
    (z0, z1), (y0, y1), (x0, x1) = bounds_um_zyx
    edges = [
        ((x0, y0, z0), (x1, y0, z0)),
        ((x1, y0, z0), (x1, y1, z0)),
        ((x1, y1, z0), (x0, y1, z0)),
        ((x0, y1, z0), (x0, y0, z0)),
        ((x0, y0, z1), (x1, y0, z1)),
        ((x1, y0, z1), (x1, y1, z1)),
        ((x1, y1, z1), (x0, y1, z1)),
        ((x0, y1, z1), (x0, y0, z1)),
        ((x0, y0, z0), (x0, y0, z1)),
        ((x1, y0, z0), (x1, y0, z1)),
        ((x1, y1, z0), (x1, y1, z1)),
        ((x0, y1, z0), (x0, y1, z1)),
    ]
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for p_a, p_b in edges:
        xs.extend([p_a[0], p_b[0], None])
        ys.extend([p_a[1], p_b[1], None])
        zs.extend([p_a[2], p_b[2], None])
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        name=name,
        line={"color": color, "width": 5},
        hoverinfo="name",
    )


def build_3d_isosurface_figure(
    run: ExecutedRun,
    *,
    clip_pct_zyx: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.0, 100.0),
        (0.0, 100.0),
        (0.0, 100.0),
    ),
    opacity: float = 0.92,
    show_coating: bool = True,
    show_roi_box: bool = True,
    max_render_side: int = 52,
    height: int = 540,
    dark_mode: bool = False,
) -> go.Figure:
    """Extract marching-cubes 3D surfaces per solid phase with interactive clip-box and ROI wireframe."""
    tok = _theme_tokens(dark_mode)
    render_vol, _ = enforce_solve_grid_cap(
        run.phase_volume, max_side=max_render_side, policy="mode_downsample"
    )
    arr = render_vol.data.copy()
    nz, ny, nx = arr.shape
    dz_um, dy_um, dx_um = render_vol.spacing_in("um")

    # Apply percentage clip-box
    (cz0, cz1), (cy0, cy1), (cx0, cx1) = clip_pct_zyx
    iz0, iz1 = int(round(nz * cz0 / 100.0)), int(round(nz * cz1 / 100.0))
    iy0, iy1 = int(round(ny * cy0 / 100.0)), int(round(ny * cy1 / 100.0))
    ix0, ix1 = int(round(nx * cx0 / 100.0)), int(round(nx * cx1 / 100.0))

    clipped = np.zeros_like(arr)
    if iz1 > iz0 and iy1 > iy0 and ix1 > ix0:
        clipped[iz0:iz1, iy0:iy1, ix0:ix1] = arr[iz0:iz1, iy0:iy1, ix0:ix1]

    traces: list[go.BaseTraceType] = []
    phase_specs = [(1, "Solid Backbone (Phase 1)", "#64748b", float(opacity))]
    if show_coating and np.any(clipped == 2):
        phase_specs.append((2, "Active Coating / CBD (Phase 2)", "#f59e0b", min(1.0, float(opacity) * 0.95)))

    for pid, pname, color, alpha in phase_specs:
        mask = (clipped == pid).astype(np.float32)
        if int(np.count_nonzero(mask)) < 4:
            continue
        # Pad by 1 voxel of zeros so exterior faces close cleanly under marching cubes
        padded = np.pad(mask, 1, mode="constant", constant_values=0.0)
        try:
            verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=(dz_um, dy_um, dx_um))
        except ValueError:
            continue
        # Shift back by 1 spacing to undo padding offset
        verts_z = verts[:, 0] - dz_um
        verts_y = verts[:, 1] - dy_um
        verts_x = verts[:, 2] - dx_um
        traces.append(
            go.Mesh3d(
                x=verts_x,
                y=verts_y,
                z=verts_z,
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                name=pname,
                color=color,
                opacity=alpha,
                flatshading=True,
                lighting={"ambient": 0.55, "diffuse": 0.75, "specular": 0.25, "roughness": 0.6},
                showlegend=True,
            )
        )

    if show_roi_box and run.summary.zoom_rois:
        roi = run.summary.zoom_rois[0]
        orig_dz, orig_dy, orig_dx = run.phase_volume.spacing_in("um")
        (rz0, rz1), (ry0, ry1), (rx0, rx1) = roi.bounds_zyx
        traces.append(
            _wireframe_box_trace(
                (
                    (rz0 * orig_dz, rz1 * orig_dz),
                    (ry0 * orig_dy, ry1 * orig_dy),
                    (rx0 * orig_dx, rx1 * orig_dx),
                ),
                name=f"Zoom ROI #1 (|J|={roi.peak_flux_magnitude:.1f}x)",
            )
        )

    if not traces:
        return empty_placeholder_figure(
            "No solid phase surfaces inside current clip box (open domain or empty clip range).",
            height=height,
            dark_mode=dark_mode,
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title={
            "text": f"3D Phase Isosurface & Synchrotron Zoom-ROI ({render_vol.shape[2]}×{render_vol.shape[1]}×{render_vol.shape[0]} render grid)",
            "font": {"size": 14},
        },
        height=height,
        paper_bgcolor=tok["paper_bg"],
        font={"color": tok["fg"]},
        margin={"l": 10, "r": 10, "t": 44, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 0.01, "x": 0.01},
        scene={
            "xaxis": {"title": "x [µm]", "backgroundcolor": tok["plot_bg"], "gridcolor": tok["grid"]},
            "yaxis": {"title": "y [µm]", "backgroundcolor": tok["plot_bg"], "gridcolor": tok["grid"]},
            "zaxis": {"title": "z [µm]", "backgroundcolor": tok["plot_bg"], "gridcolor": tok["grid"]},
            "aspectmode": "data",
        },
    )
    return fig


def build_gp_steering_figure(
    suggestion: BayesianSetpointSuggestion,
    *,
    height: int = 490,
    dark_mode: bool = False,
) -> go.Figure:
    """Build interactive two-panel Gaussian Process posterior + Expected Improvement plot."""
    tok = _theme_tokens(dark_mode)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.66, 0.34],
        vertical_spacing=0.10,
        subplot_titles=(
            f"GP Surrogate: Cell Resistance R_cell vs {suggestion.parameter_name}",
            "Expected Improvement Acquisition EI(σ)",
        ),
    )

    gx = suggestion.grid_x
    fig.add_trace(
        go.Scatter(
            x=gx + gx[::-1],
            y=suggestion.gp_upper_95 + suggestion.gp_lower_95[::-1],
            fill="toself",
            fillcolor="rgba(59, 130, 246, 0.22)",
            line={"color": "rgba(255,255,255,0)"},
            name="GP 95% Credible Band",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=gx,
            y=suggestion.gp_mean,
            mode="lines",
            name="GP Posterior Mean R_cell",
            line={"color": "#3b82f6", "width": 2.6},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=suggestion.observed_x,
            y=suggestion.observed_r_cell,
            mode="markers",
            name="Executed OpenImpala Solves",
            marker={"size": 11, "color": "#f59e0b", "line": {"width": 1.5, "color": tok["fg"]}},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=gx,
            y=suggestion.ei_curve,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.25)",
            line={"color": "#10b981", "width": 2.2},
            name="Expected Improvement",
        ),
        row=2,
        col=1,
    )

    for r in (1, 2):
        fig.add_vline(
            x=suggestion.suggested_value,
            line_dash="dash",
            line_color="#10b981",
            line_width=2.0,
            row=r,
            col=1,
        )

    fig.update_layout(
        height=height,
        paper_bgcolor=tok["paper_bg"],
        plot_bgcolor=tok["plot_bg"],
        font={"color": tok["fg"]},
        margin={"l": 56, "r": 28, "t": 48, "b": 46},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.05, "x": 0.0},
    )
    fig.update_xaxes(gridcolor=tok["grid"])
    fig.update_yaxes(gridcolor=tok["grid"])
    fig.update_yaxes(title_text="R_cell [-]", row=1, col=1)
    fig.update_yaxes(title_text="EI [-]", row=2, col=1)
    fig.update_xaxes(
        title_text=f"{suggestion.parameter_name} [{suggestion.parameter_unit}]",
        row=2,
        col=1,
    )
    return fig
