"""Declarative Dash layout for The OpenImpala Hub."""

from __future__ import annotations

from dash import dcc, html

from .store import RunRegistry


def _initial_history_records(registry: RunRegistry) -> list[dict]:
    records: list[dict] = []
    for rid in registry.boot_run_ids:
        run = registry.get(rid)
        if run is None:
            continue
        s = run.summary
        tz = s.directional.get("z")
        records.append(
            {
                "run_id": rid,
                "label": s.label,
                "clamping_kpa": float(s.microstructure_spec.get("clamping_kpa", 150.0)),
                "porosity": s.porosity,
                "tau_z": tz.tortuosity if tz else None,
                "d_eff_z": tz.effective_diffusivity if tz else 0.0,
                "sigma_z": s.solid_effective_conductivity_z,
                "r_cell": s.cell_resistance_fom,
                "source_kind": s.source_kind,
            }
        )
    return records


def build_hub_layout(registry: RunRegistry) -> html.Div:
    """Construct the root Dash component tree seeded with warm-start run IDs."""
    init_history = _initial_history_records(registry)
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
                    # Left Sidebar: Step 1 (Source) + Step 2 (Physics Solver)
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
                                        options=[
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
                                            {
                                                "label": "PorousDiff 3D Diffusion Gallery (.ome.zarr)",
                                                "value": "porousdiff_gallery",
                                            },
                                            {
                                                "label": "Upload Volume (.tif, .npy, .h5)",
                                                "value": "user_upload",
                                            },
                                        ],
                                    ),
                                ]
                            ),
                            html.Div(
                                id="upload-container",
                                style={"display": "none"},
                                children=[
                                    dcc.Upload(
                                        id="upload-volume-box",
                                        className="upload-box",
                                        children=html.Div(
                                            ["Drag & drop or ", html.B("select .tif / .npy / .h5")]
                                        ),
                                        multiple=False,
                                    ),
                                    html.Div(
                                        id="upload-status-text",
                                        style={"fontSize": "11.5px", "marginTop": "4px"},
                                    ),
                                ],
                            ),
                            html.Div(
                                children=[
                                    html.Label("Cubic Grid Resolution (N³)", className="field-label"),
                                    dcc.Slider(
                                        id="grid-side-slider",
                                        min=32,
                                        max=96,
                                        step=16,
                                        value=48,
                                        marks={32: "32³", 48: "48³", 64: "64³", 80: "80³", 96: "96³"},
                                    ),
                                ]
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
                                        min=0.05,
                                        max=100.0,
                                        step="any",
                                        className="num-input",
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
                                    html.Label("Target / Base Porosity ε [-]", className="field-label"),
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
                                children=[
                                    html.Label("Deterministic Random Seed", className="field-label"),
                                    dcc.Input(
                                        id="seed-input",
                                        type="number",
                                        value=42,
                                        min=0,
                                        step=1,
                                        className="num-input",
                                    ),
                                ]
                            ),
                            html.Hr(style={"width": "100%", "borderColor": "var(--border-color)"}),
                            html.Div("Step 2 · OpenImpala Physics", className="sidebar-heading"),
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
                                    }
                                ],
                                value=[],
                                style={"fontSize": "12.5px"},
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
                                                                            {"label": " XZ (Flow)", "value": "xz"},
                                                                            {"label": " XY (In-Plane)", "value": "xy"},
                                                                            {"label": " YZ (Cross)", "value": "yz"},
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
                                                                style={"flex": "1", "minWidth": "200px"},
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
                                                            dcc.Graph(id="directional-profile-graph"),
                                                        ],
                                                    ),
                                                    html.Details(
                                                        style={"marginTop": "10px"},
                                                        children=[
                                                            html.Summary(
                                                                "Continuum PyBaMM Parameters & Homogenised Tensor Readout",
                                                                style={"cursor": "pointer", "fontWeight": "600", "fontSize": "13px"},
                                                            ),
                                                            html.Pre(id="pybamm-bpx-preview", className="code-preview"),
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
                                                                    html.Span("Surface Opacity", className="field-label"),
                                                                    dcc.Slider(
                                                                        id="opacity-3d-slider",
                                                                        min=0.25,
                                                                        max=1.0,
                                                                        step=0.05,
                                                                        value=0.90,
                                                                        marks={0.3: "0.3", 0.65: "0.65", 1.0: "1.0"},
                                                                        updatemode="mouseup",
                                                                    ),
                                                                ],
                                                            ),
                                                            html.Div(
                                                                style={"minWidth": "170px"},
                                                                children=[
                                                                    html.Span("Z Cutaway [%]", className="field-label"),
                                                                    dcc.RangeSlider(
                                                                        id="clip-z-slider",
                                                                        min=0,
                                                                        max=100,
                                                                        step=5,
                                                                        value=[0, 100],
                                                                        marks={0: "0%", 50: "50%", 100: "100%"},
                                                                        updatemode="mouseup",
                                                                    ),
                                                                ],
                                                            ),
                                                            html.Div(
                                                                style={"minWidth": "170px"},
                                                                children=[
                                                                    html.Span("X Cutaway [%]", className="field-label"),
                                                                    dcc.RangeSlider(
                                                                        id="clip-x-slider",
                                                                        min=0,
                                                                        max=100,
                                                                        step=5,
                                                                        value=[0, 100],
                                                                        marks={0: "0%", 50: "50%", 100: "100%"},
                                                                        updatemode="mouseup",
                                                                    ),
                                                                ],
                                                            ),
                                                        ],
                                                    ),
                                                    dcc.Loading(
                                                        type="default",
                                                        children=dcc.Graph(id="isosurface-3d-graph"),
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
                                                                    html.Span("Left Run", className="field-label"),
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
                                                                    html.Span("Right Run", className="field-label"),
                                                                    dcc.Dropdown(
                                                                        id="compare-right-dropdown",
                                                                        clearable=False,
                                                                        value=cmp_right_id,
                                                                    ),
                                                                ],
                                                            ),
                                                            html.Div(
                                                                children=[
                                                                    html.Span("Field", className="field-label"),
                                                                    dcc.RadioItems(
                                                                        id="compare-layer-radio",
                                                                        options=[
                                                                            {"label": " Flux |J/J0|", "value": "flux"},
                                                                            {"label": " Phase", "value": "phase"},
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
                                                    html.Div(id="compare-delta-table", style={"marginTop": "10px"}),
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
                                                            html.Button(
                                                                "⚡ Execute Recommended Scan in OpenImpala",
                                                                id="run-suggested-setpoint-btn",
                                                                className="btn btn-emerald",
                                                                n_clicks=0,
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
                                                                        style={"marginTop": "0", "fontSize": "14px"},
                                                                    ),
                                                                    html.Div(id="zoom-roi-cards"),
                                                                    html.H4(
                                                                        "Evaluated Compression Campaign History",
                                                                        style={"marginTop": "18px", "fontSize": "14px"},
                                                                    ),
                                                                    html.Div(id="sweep-history-table"),
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
