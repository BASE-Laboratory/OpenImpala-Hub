"""Declarative Dash layout for The OpenImpala Hub (`v0.2`)."""

from __future__ import annotations

from dash import dcc, html

from ..engine.generators import list_porousdiff_gallery
from .store import RunRegistry


def history_entry_from_summary(s) -> dict:
    """Build a compact JSON-serializable campaign history dict preserving live regime ingredients."""
    tz = s.directional.get("z")
    mspec = s.microstructure_spec or {}
    strain = float(mspec.get("compression_strain", 0.0))
    thickness_ratio = (
        max(0.35, 1.0 - strain)
        if s.source_kind in ("carbon_felt", "porousdiff_gallery")
        else 1.0
    )
    return {
        "run_id": s.run_id,
        "label": s.label,
        "clamping_kpa": float(mspec.get("clamping_kpa", 150.0)),
        "porosity": s.porosity,
        "tau_z": tz.tortuosity if tz else None,
        "d_eff_z": tz.effective_diffusivity if tz else s.porosity,
        "sigma_z": s.solid_effective_conductivity_z,
        "active_ssa": s.interfacial_area_active_m_inv,
        "thickness_ratio": round(thickness_ratio, 4),
        "r_cell": s.cell_resistance_fom,
        "source_kind": s.source_kind,
        "microstructure_spec": mspec,
        "solve_spec": s.solve_spec,
    }


def initial_history_records(registry: RunRegistry) -> list[dict]:
    records: list[dict] = []
    for rid in registry.boot_run_ids:
        run = registry.get(rid)
        if run is None:
            continue
        records.append(history_entry_from_summary(run.summary))
    return records


def _source_dropdown_options() -> list[dict]:
    gallery = list_porousdiff_gallery()
    if gallery:
        pd_opt = {
            "label": f"PorousDiff 3D Diffusion Gallery ({len(gallery)} .ome.zarr staged)",
            "value": "porousdiff_gallery",
        }
    else:
        pd_opt = {
            "label": "PorousDiff 3D Diffusion (0 staged — drop .ome.zarr in data/porousdiff)",
            "value": "porousdiff_gallery",
            "disabled": True,
        }
    return [
        {
            "label": "Anisotropic Carbon Felt (Fibrous + Coating)",
            "value": "carbon_felt",
        },
        {
            "label": "3-Phase Composite Cathode (NMC + CBD)",
            "value": "multiphase_electrode",
        },
        {
            "label": "Anisotropic Gaussian Random Blobs",
            "value": "gaussian_blobs",
        },
        {
            "label": "Overlapping Sphere Packing",
            "value": "overlapping_spheres",
        },
        {
            "label": "Benchmark XCT / Sphere Pack (openimpala.datasets)",
            "value": "benchmark_dataset",
        },
        {
            "label": "Analytical Open Cube (τ = 1.000 Verification)",
            "value": "open_cube",
        },
        pd_opt,
        {
            "label": "Upload Experimental Volume (.tif, .npy, .h5, .ome.zarr.zip)",
            "value": "user_upload",
        },
    ]


def build_hub_layout(registry: RunRegistry) -> html.Div:
    """Construct the root Dash component tree seeded with warm-start run IDs."""
    init_history = initial_history_records(registry)
    default_id = registry.default_run_id or (init_history[0]["run_id"] if init_history else None)
    cmp_left_id = init_history[0]["run_id"] if init_history else default_id
    cmp_right_id = init_history[-1]["run_id"] if len(init_history) > 1 else default_id

    return html.Div(
        id="hub-theme-wrapper",
        className="hub-root",
        **{"data-theme": "light"},
        children=[
            dcc.Store(id="active-run-id-store", data=default_id),
            dcc.Store(id="run-history-store", data=init_history),
            dcc.Store(id="theme-store", data="light", storage_type="local"),
            dcc.Store(id="uploaded-file-store", data=None),
            dcc.Store(id="suggested-kpa-store", data=245.0),
            dcc.Download(id="download-bundle"),
            dcc.Download(id="download-bpx"),
            # ── Top Navigation & Preset Bar ──
            html.Header(
                className="hub-topbar",
                children=[
                    html.Div(
                        className="brand-block",
                        children=[
                            html.Span("BASE LAB", className="brand-badge"),
                            html.Div(
                                children=[
                                    html.H1("The OpenImpala Hub", className="brand-title"),
                                    html.P(
                                        "3D Microstructure Synthesis · AMReX Transport Physics · Autonomous Experiment Steering",
                                        className="brand-subtitle",
                                    ),
                                ]
                            ),
                        ],
                    ),
                    html.Div(
                        className="preset-bar",
                        children=[
                            html.Span(
                                "1-Click Presets:",
                                style={"fontSize": "12px", "fontWeight": "600"},
                            ),
                            html.Button(
                                "1 · Carbon Felt Compression",
                                id="preset-btn-carbon",
                                className="btn",
                                n_clicks=0,
                            ),
                            html.Button(
                                "2 · 3-Phase Battery Cathode",
                                id="preset-btn-cathode",
                                className="btn",
                                n_clicks=0,
                            ),
                            html.Button(
                                "3a · Teaching: Cube (τ=1.000)",
                                id="preset-btn-cube",
                                className="btn",
                                n_clicks=0,
                            ),
                            html.Button(
                                "3b · Benchmark Spheres",
                                id="preset-btn-spheres",
                                className="btn",
                                n_clicks=0,
                            ),
                        ],
                    ),
                    html.Div(
                        className="action-bar",
                        children=[
                            html.Button(
                                "⬇ Export NIA Bundle (.zip)",
                                id="export-bundle-btn",
                                className="btn btn-emerald",
                                n_clicks=0,
                            ),
                            html.Button(
                                "⬇ BPX JSON",
                                id="export-bpx-btn",
                                className="btn",
                                n_clicks=0,
                            ),
                            html.Button(
                                "🌙 Dark / Light",
                                id="theme-toggle-btn",
                                className="btn",
                                n_clicks=0,
                            ),
                        ],
                    ),
                ],
            ),
            # ── Main Body Grid ──
            html.Div(
                className="hub-body",
                children=[
                    # Left Sidebar: Step 1 (Source & Segmentation) + Step 2 (Physics Solver)
                    html.Aside(
                        className="hub-sidebar",
                        children=[
                            html.Div("Step 1 · Microstructure Source", className="sidebar-heading"),
                            html.Div(
                                children=[
                                    html.Label("Generator / Dataset", className="field-label"),
                                    dcc.Dropdown(
                                        id="source-kind-dropdown",
                                        clearable=False,
                                        value="carbon_felt",
                                        options=_source_dropdown_options(),
                                    ),
                                ]
                            ),
                            # Experimental Upload & Pre-Solve Segmentation Drawer
                            html.Div(
                                id="upload-container",
                                style={"display": "none"},
                                children=[
                                    dcc.Upload(
                                        id="upload-volume-box",
                                        className="upload-box",
                                        children=html.Div(
                                            [
                                                "Drag & drop or ",
                                                html.B("select .tif / .npy / .h5 / .ome.zarr.zip"),
                                            ]
                                        ),
                                        multiple=False,
                                    ),
                                    html.Div(
                                        id="upload-status-text",
                                        style={"fontSize": "11.5px", "marginTop": "4px"},
                                    ),
                                    html.Div(
                                        id="upload-seg-drawer",
                                        className="upload-drawer",
                                        style={"display": "none"},
                                        children=[
                                            html.Div(
                                                children=[
                                                    html.Label(
                                                        "Segmentation Mode",
                                                        className="field-label",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="upload-seg-mode-dropdown",
                                                        clearable=False,
                                                        value="otsu",
                                                        options=[
                                                            {
                                                                "label": "Otsu Automatic Threshold",
                                                                "value": "otsu",
                                                            },
                                                            {
                                                                "label": "Manual Threshold",
                                                                "value": "manual",
                                                            },
                                                            {
                                                                "label": "Already Integer Phase IDs",
                                                                "value": "labels",
                                                            },
                                                        ],
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                id="upload-manual-threshold-block",
                                                style={"display": "none"},
                                                children=[
                                                    html.Label(
                                                        "Threshold Value",
                                                        className="field-label",
                                                    ),
                                                    dcc.Input(
                                                        id="upload-threshold-input",
                                                        type="number",
                                                        value=0.5,
                                                        step="any",
                                                        className="num-input",
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                children=[
                                                    html.Label(
                                                        "Contrast Polarity",
                                                        className="field-label",
                                                    ),
                                                    dcc.RadioItems(
                                                        id="upload-polarity-radio",
                                                        options=[
                                                            {
                                                                "label": " Dark = Pore (0)",
                                                                "value": "dark",
                                                            },
                                                            {
                                                                "label": " Bright = Pore (0)",
                                                                "value": "bright",
                                                            },
                                                        ],
                                                        value="dark",
                                                        inline=True,
                                                        style={"fontSize": "12px"},
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                children=[
                                                    html.Label(
                                                        "Large-Volume Cap Policy (>128³)",
                                                        className="field-label",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="upload-coarsen-dropdown",
                                                        clearable=False,
                                                        value="crop_center",
                                                        options=[
                                                            {
                                                                "label": "Centre ROI Crop (preserve dx)",
                                                                "value": "crop_center",
                                                            },
                                                            {
                                                                "label": "Majority-Mode Coarsen (full FoV)",
                                                                "value": "mode_downsample",
                                                            },
                                                        ],
                                                    ),
                                                ]
                                            ),
                                            dcc.Graph(
                                                id="upload-preview-graph",
                                                config={"displayModeBar": False},
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                id="grid-side-block",
                                children=[
                                    html.Label(
                                        "Cubic Grid Resolution (N³)",
                                        className="field-label",
                                    ),
                                    dcc.Slider(
                                        id="grid-side-slider",
                                        min=32,
                                        max=96,
                                        step=16,
                                        value=48,
                                        marks={
                                            32: "32³",
                                            48: "48³",
                                            64: "64³",
                                            80: "80³",
                                            96: "96³",
                                        },
                                    ),
                                ],
                            ),
                            html.Div(
                                children=[
                                    html.Label(
                                        "Physical Voxel Spacing [µm] (SI calibrated)",
                                        className="field-label",
                                    ),
                                    dcc.Input(
                                        id="voxel-size-input",
                                        type="number",
                                        value=1.6,
                                        min=0.005,
                                        max=500.0,
                                        step="any",
                                        className="num-input",
                                    ),
                                    html.Details(
                                        id="aniso-spacing-details",
                                        style={"marginTop": "5px"},
                                        children=[
                                            html.Summary(
                                                "Anisotropic Spacing (dz, dy, dx) [µm]",
                                                style={
                                                    "fontSize": "11.5px",
                                                    "cursor": "pointer",
                                                    "color": "var(--text-muted)",
                                                },
                                            ),
                                            html.Div(
                                                className="three-col-inputs",
                                                children=[
                                                    html.Div(
                                                        [
                                                            html.Span(
                                                                "dz [µm]",
                                                                style={"fontSize": "11px"},
                                                            ),
                                                            dcc.Input(
                                                                id="voxel-dz-input",
                                                                type="number",
                                                                value=1.6,
                                                                min=0.005,
                                                                step="any",
                                                                className="num-input",
                                                            ),
                                                        ]
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Span(
                                                                "dy [µm]",
                                                                style={"fontSize": "11px"},
                                                            ),
                                                            dcc.Input(
                                                                id="voxel-dy-input",
                                                                type="number",
                                                                value=1.6,
                                                                min=0.005,
                                                                step="any",
                                                                className="num-input",
                                                            ),
                                                        ]
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Span(
                                                                "dx [µm]",
                                                                style={"fontSize": "11px"},
                                                            ),
                                                            dcc.Input(
                                                                id="voxel-dx-input",
                                                                type="number",
                                                                value=1.6,
                                                                min=0.005,
                                                                step="any",
                                                                className="num-input",
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ]
                            ),
                            html.Div(
                                id="clamping-control-block",
                                children=[
                                    html.Label(
                                        "Clamping Pressure σ [kPa] (Z-compression)",
                                        className="field-label",
                                    ),
                                    dcc.Slider(
                                        id="clamping-kpa-slider",
                                        min=20,
                                        max=500,
                                        step=10,
                                        value=150,
                                        marks={20: "20", 150: "150", 300: "300", 500: "500"},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                    ),
                                ],
                            ),
                            html.Div(
                                id="porosity-control-block",
                                children=[
                                    html.Label(
                                        "Target / Base Porosity ε [-]",
                                        className="field-label",
                                    ),
                                    dcc.Slider(
                                        id="porosity-slider",
                                        min=0.20,
                                        max=0.88,
                                        step=0.02,
                                        value=0.84,
                                        marks={0.2: "0.20", 0.5: "0.50", 0.84: "0.84"},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                    ),
                                ],
                            ),
                            html.Div(
                                id="seed-control-block",
                                children=[
                                    html.Label(
                                        "Deterministic Random Seed",
                                        className="field-label",
                                    ),
                                    dcc.Input(
                                        id="seed-input",
                                        type="number",
                                        value=42,
                                        min=0,
                                        step=1,
                                        className="num-input",
                                    ),
                                ],
                            ),
                            html.Hr(style={"width": "100%", "borderColor": "var(--border-color)"}),
                            html.Div("Step 2 · OpenImpala Physics", className="sidebar-heading"),
                            html.Div(
                                children=[
                                    html.Label("Conducting Phase ID", className="field-label"),
                                    dcc.Dropdown(
                                        id="conducting-phase-dropdown",
                                        clearable=False,
                                        value=0,
                                        options=[
                                            {
                                                "label": "Phase 0: Pore / Electrolyte",
                                                "value": 0,
                                            },
                                            {
                                                "label": "Phase 1: Solid Matrix / Backbone",
                                                "value": 1,
                                            },
                                        ],
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Label(
                                        "Cell Role (PyBaMM / Faraday BPX)",
                                        className="field-label",
                                    ),
                                    dcc.Dropdown(
                                        id="electrode-role-dropdown",
                                        clearable=False,
                                        value="negative",
                                        options=[
                                            {
                                                "label": "Negative Electrode (Anode / Felt)",
                                                "value": "negative",
                                            },
                                            {
                                                "label": "Positive Electrode (Cathode)",
                                                "value": "positive",
                                            },
                                            {
                                                "label": "Porous Separator",
                                                "value": "separator",
                                            },
                                        ],
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Label("Linear Solver Backend", className="field-label"),
                                    dcc.Dropdown(
                                        id="solver-backend-dropdown",
                                        clearable=False,
                                        value="mlmg",
                                        options=[
                                            {
                                                "label": "AMReX Matrix-Free MLMG (C++)",
                                                "value": "mlmg",
                                            },
                                            {
                                                "label": "HYPRE Krylov + PFMG (C++)",
                                                "value": "hypre",
                                            },
                                            {
                                                "label": "Pure-Python SciPy CG (Fallback)",
                                                "value": "scipy",
                                            },
                                        ],
                                    ),
                                ]
                            ),
                            dcc.Checklist(
                                id="homogenisation-check",
                                options=[
                                    {
                                        "label": " Compute 3×3 Periodic D_eff Tensor",
                                        "value": "homog",
                                    },
                                    {
                                        "label": " Compute Subvolume REV Curve (oi.rev_study)",
                                        "value": "rev",
                                    },
                                ],
                                value=[],
                                style={
                                    "fontSize": "12.5px",
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "gap": "4px",
                                },
                            ),
                            dcc.Loading(
                                type="circle",
                                children=html.Div(
                                    children=[
                                        html.Button(
                                            "▶ Generate & Run OpenImpala",
                                            id="run-pipeline-btn",
                                            className="btn btn-primary",
                                            n_clicks=0,
                                        ),
                                        html.Div(id="run-error-banner", style={"marginTop": "8px"}),
                                    ]
                                ),
                            ),
                        ],
                    ),
                    # Right Main Workspace
                    html.Main(
                        className="hub-main",
                        children=[
                            html.Div(id="preset-context-banner", className="context-banner"),
                            html.Div(id="kpi-cards-row", className="kpi-grid"),
                            dcc.Tabs(
                                id="main-tabs",
                                value="tab-2d",
                                children=[
                                    # Tab 1: 2D Slices & Transport Flux
                                    dcc.Tab(
                                        label="2D Slices & Transport Flux",
                                        value="tab-2d",
                                        children=[
                                            html.Div(
                                                className="panel-card",
                                                children=[
                                                    html.Div(
                                                        className="controls-strip",
                                                        children=[
                                                            html.Div(
                                                                children=[
                                                                    html.Span(
                                                                        "Plane: ",
                                                                        className="field-label",
                                                                    ),
                                                                    dcc.RadioItems(
                                                                        id="slice-plane-radio",
                                                                        options=[
                                                                            {
                                                                                "label": " XZ (Flow)",
                                                                                "value": "xz",
                                                                            },
                                                                            {
                                                                                "label": " XY (In-Plane)",
                                                                                "value": "xy",
                                                                            },
                                                                            {
                                                                                "label": " YZ (Cross)",
                                                                                "value": "yz",
                                                                            },
                                                                        ],
                                                                        value="xz",
                                                                        inline=True,
                                                                        style={"fontSize": "13px"},
                                                                    ),
                                                                ]
                                                            ),
                                                            html.Div(
                                                                children=[
                                                                    html.Span(
                                                                        "Field Overlay: ",
                                                                        className="field-label",
                                                                    ),
                                                                    dcc.RadioItems(
                                                                        id="slice-layer-radio",
                                                                        options=[
                                                                            {
                                                                                "label": " Diffusive Flux |J/J0|",
                                                                                "value": "flux",
                                                                            },
                                                                            {
                                                                                "label": " Potential φ",
                                                                                "value": "potential",
                                                                            },
                                                                            {
                                                                                "label": " Phase Labels",
                                                                                "value": "phase",
                                                                            },
                                                                        ],
                                                                        value="flux",
                                                                        inline=True,
                                                                        style={"fontSize": "13px"},
                                                                    ),
                                                                ]
                                                            ),
                                                            html.Div(
                                                                style={
                                                                    "flex": "1",
                                                                    "minWidth": "200px",
                                                                },
                                                                children=[
                                                                    html.Span(
                                                                        "Slice Index",
                                                                        className="field-label",
                                                                    ),
                                                                    dcc.Slider(
                                                                        id="slice-index-slider",
                                                                        min=0,
                                                                        max=47,
                                                                        step=1,
                                                                        value=24,
                                                                        marks=None,
                                                                        tooltip={
                                                                            "placement": "bottom",
                                                                            "always_visible": True,
                                                                        },
                                                                    ),
                                                                ],
                                                            ),
                                                        ],
                                                    ),
                                                    html.Div(
                                                        className="two-col-charts",
                                                        children=[
                                                            dcc.Graph(id="slice-heatmap-graph"),
                                                            dcc.Graph(
                                                                id="directional-profile-graph"
                                                            ),
                                                        ],
                                                    ),
                                                    html.Details(
                                                        style={"marginTop": "10px"},
                                                        children=[
                                                            html.Summary(
                                                                "Continuum PyBaMM Parameters & Homogenised Tensor Readout",
                                                                style={
                                                                    "cursor": "pointer",
                                                                    "fontWeight": "600",
                                                                    "fontSize": "13px",
                                                                },
                                                            ),
                                                            html.Pre(
                                                                id="pybamm-bpx-preview",
                                                                className="code-preview",
                                                            ),
                                                        ],
                                                    ),
                                                ],
                                            )
                                        ],
                                    ),
                                    # Tab 2: 3D Volumetric Cutaway
                                    dcc.Tab(
                                        label="3D Microstructure & Cutaway",
                                        value="tab-3d",
                                        children=[
                                            html.Div(
                                                className="panel-card",
                                                children=[
                                                    html.Div(
                                                        className="controls-strip",
                                                        children=[
                                                            dcc.Checklist(
                                                                id="options-3d-check",
                                                                options=[
                                                                    {
                                                                        "label": " Show Active Coating (Phase 2)",
                                                                        "value": "coating",
                                                                    },
                                                                    {
                                                                        "label": " Show Synchrotron Zoom-ROI #1 Box",
                                                                        "value": "roi",
                                                                    },
                                                                ],
                                                                value=["coating", "roi"],
                                                                inline=True,
                                                                style={"fontSize": "13px"},
                                                            ),
                                                            html.Div(
                                                                style={"minWidth": "150px"},
                                                                children=[
                                                                    html.Span(
                                                                        "Surface Opacity",
                                                                        className="field-label",
                                                                    ),
                                                                    dcc.Slider(
                                                                        id="opacity-3d-slider",
                                                                        min=0.25,
                                                                        max=1.0,
                                                                        step=0.05,
                                                                        value=0.90,
                                                                        marks={
                                                                            0.3: "0.3",
                                                                            0.65: "0.65",
                                                                            1.0: "1.0",
                                                                        },
                                                                        updatemode="mouseup",
                                                                    ),
                                                                ],
                                                            ),
                                                            html.Div(
                                                                style={"minWidth": "170px"},
                                                                children=[
                                                                    html.Span(
                                                                        "Z Cutaway [%]",
                                                                        className="field-label",
                                                                    ),
                                                                    dcc.RangeSlider(
                                                                        id="clip-z-slider",
                                                                        min=0,
                                                                        max=100,
                                                                        step=5,
                                                                        value=[0, 100],
                                                                        marks={
                                                                            0: "0%",
                                                                            50: "50%",
                                                                            100: "100%",
                                                                        },
                                                                        updatemode="mouseup",
                                                                    ),
                                                                ],
                                                            ),
                                                            html.Div(
                                                                style={"minWidth": "170px"},
                                                                children=[
                                                                    html.Span(
                                                                        "X Cutaway [%]",
                                                                        className="field-label",
                                                                    ),
                                                                    dcc.RangeSlider(
                                                                        id="clip-x-slider",
                                                                        min=0,
                                                                        max=100,
                                                                        step=5,
                                                                        value=[0, 100],
                                                                        marks={
                                                                            0: "0%",
                                                                            50: "50%",
                                                                            100: "100%",
                                                                        },
                                                                        updatemode="mouseup",
                                                                    ),
                                                                ],
                                                            ),
                                                        ],
                                                    ),
                                                    dcc.Loading(
                                                        type="default",
                                                        children=dcc.Graph(
                                                            id="isosurface-3d-graph"
                                                        ),
                                                    ),
                                                ],
                                            )
                                        ],
                                    ),
                                    # Tab 3: Side-by-Side State Comparison
                                    dcc.Tab(
                                        label="Compare States",
                                        value="tab-compare",
                                        children=[
                                            html.Div(
                                                className="panel-card",
                                                children=[
                                                    html.Div(
                                                        className="controls-strip",
                                                        children=[
                                                            html.Div(
                                                                style={"minWidth": "240px"},
                                                                children=[
                                                                    html.Span(
                                                                        "Left Run",
                                                                        className="field-label",
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="compare-left-dropdown",
                                                                        clearable=False,
                                                                        value=cmp_left_id,
                                                                    ),
                                                                ],
                                                            ),
                                                            html.Div(
                                                                style={"minWidth": "240px"},
                                                                children=[
                                                                    html.Span(
                                                                        "Right Run",
                                                                        className="field-label",
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="compare-right-dropdown",
                                                                        clearable=False,
                                                                        value=cmp_right_id,
                                                                    ),
                                                                ],
                                                            ),
                                                            html.Div(
                                                                children=[
                                                                    html.Span(
                                                                        "Field",
                                                                        className="field-label",
                                                                    ),
                                                                    dcc.RadioItems(
                                                                        id="compare-layer-radio",
                                                                        options=[
                                                                            {
                                                                                "label": " Flux |J/J0|",
                                                                                "value": "flux",
                                                                            },
                                                                            {
                                                                                "label": " Phase",
                                                                                "value": "phase",
                                                                            },
                                                                        ],
                                                                        value="flux",
                                                                        inline=True,
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                    ),
                                                    html.Div(
                                                        className="equal-two-col",
                                                        children=[
                                                            dcc.Graph(id="compare-left-graph"),
                                                            dcc.Graph(id="compare-right-graph"),
                                                        ],
                                                    ),
                                                    html.Div(
                                                        id="compare-delta-table",
                                                        style={"marginTop": "10px"},
                                                    ),
                                                ],
                                            )
                                        ],
                                    ),
                                    # Tab 4: WP3 Autonomous Experiment Steering
                                    dcc.Tab(
                                        label="Autonomous Planner (Digital Twin)",
                                        value="tab-twin",
                                        children=[
                                            html.Div(
                                                className="panel-card",
                                                children=[
                                                    html.Div(
                                                        className="context-banner",
                                                        style={"marginBottom": "12px"},
                                                        children=[
                                                            html.Div(
                                                                id="twin-recommendation-banner",
                                                                style={"flex": "1"},
                                                            ),
                                                            html.Div(
                                                                style={
                                                                    "display": "flex",
                                                                    "flexWrap": "wrap",
                                                                    "gap": "8px",
                                                                },
                                                                children=[
                                                                    html.Button(
                                                                        "⚡ Execute Recommended Scan",
                                                                        id="run-suggested-setpoint-btn",
                                                                        className="btn btn-emerald",
                                                                        n_clicks=0,
                                                                    ),
                                                                    html.Button(
                                                                        "🎯 View ROI #1 in 2D Slice",
                                                                        id="jump-roi-btn",
                                                                        className="btn",
                                                                        n_clicks=0,
                                                                    ),
                                                                    html.Button(
                                                                        "↺ Reset Campaign",
                                                                        id="reset-campaign-btn",
                                                                        className="btn",
                                                                        n_clicks=0,
                                                                    ),
                                                                ],
                                                            ),
                                                        ],
                                                    ),
                                                    # Interactive Cell Regime Weights (recomputes GP live without PDE re-solve)
                                                    html.Div(
                                                        className="regime-grid",
                                                        children=[
                                                            html.Div(
                                                                children=[
                                                                    html.Span(
                                                                        "Ionic Mass-Transport Weight w_ion",
                                                                        className="field-label",
                                                                    ),
                                                                    dcc.Slider(
                                                                        id="twin-w-ion-slider",
                                                                        min=0.10,
                                                                        max=1.20,
                                                                        step=0.05,
                                                                        value=0.58,
                                                                        marks={
                                                                            0.1: "0.1",
                                                                            0.58: "0.58",
                                                                            1.2: "1.2",
                                                                        },
                                                                        tooltip={
                                                                            "placement": "bottom",
                                                                            "always_visible": False,
                                                                        },
                                                                    ),
                                                                ]
                                                            ),
                                                            html.Div(
                                                                children=[
                                                                    html.Span(
                                                                        "Electronic Contact Weight w_elec",
                                                                        className="field-label",
                                                                    ),
                                                                    dcc.Slider(
                                                                        id="twin-w-elec-slider",
                                                                        min=0.05,
                                                                        max=1.00,
                                                                        step=0.05,
                                                                        value=0.27,
                                                                        marks={
                                                                            0.05: "0.05",
                                                                            0.27: "0.27",
                                                                            1.0: "1.0",
                                                                        },
                                                                        tooltip={
                                                                            "placement": "bottom",
                                                                            "always_visible": False,
                                                                        },
                                                                    ),
                                                                ]
                                                            ),
                                                            html.Div(
                                                                children=[
                                                                    html.Span(
                                                                        "Interfacial Kinetic Weight w_kin",
                                                                        className="field-label",
                                                                    ),
                                                                    dcc.Slider(
                                                                        id="twin-w-kin-slider",
                                                                        min=0.00,
                                                                        max=0.80,
                                                                        step=0.05,
                                                                        value=0.15,
                                                                        marks={
                                                                            0.0: "0.0",
                                                                            0.15: "0.15",
                                                                            0.8: "0.8",
                                                                        },
                                                                        tooltip={
                                                                            "placement": "bottom",
                                                                            "always_visible": False,
                                                                        },
                                                                    ),
                                                                ]
                                                            ),
                                                            html.Div(
                                                                children=[
                                                                    html.Span(
                                                                        "Bulk Ratio σ_0 / κ_0",
                                                                        className="field-label",
                                                                    ),
                                                                    dcc.Slider(
                                                                        id="twin-sigma-kappa-slider",
                                                                        min=2.0,
                                                                        max=50.0,
                                                                        step=2.0,
                                                                        value=16.0,
                                                                        marks={
                                                                            2: "2",
                                                                            16: "16",
                                                                            50: "50",
                                                                        },
                                                                        tooltip={
                                                                            "placement": "bottom",
                                                                            "always_visible": False,
                                                                        },
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                    ),
                                                    html.Div(
                                                        className="two-col-charts",
                                                        children=[
                                                            dcc.Graph(id="gp-steering-graph"),
                                                            html.Div(
                                                                children=[
                                                                    html.H4(
                                                                        "Spatial Beamline Trigger · Top 3D Bottleneck ROIs",
                                                                        style={
                                                                            "marginTop": "0",
                                                                            "fontSize": "14px",
                                                                        },
                                                                    ),
                                                                    html.Div(id="zoom-roi-cards"),
                                                                    html.H4(
                                                                        "Evaluated Campaign History (Live R_cell)",
                                                                        style={
                                                                            "marginTop": "18px",
                                                                            "fontSize": "14px",
                                                                        },
                                                                    ),
                                                                    html.Div(
                                                                        id="sweep-history-table"
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                    ),
                                                ],
                                            )
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
