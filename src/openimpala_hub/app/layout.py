"""Declarative Dash layout for The OpenImpala Hub (`v0.4.0` — BASE Lab Signature)."""

from __future__ import annotations

from dash import dcc, html

from ..engine.chunked_io import stack_dimensions
from ..engine.electrochem import CHEMISTRIES
from ..engine.generators import list_porousdiff_gallery
from ..engine.mechanics import has_sms_backend
from ..engine.operando_demo import discover_operando_datasets
from .store import RunRegistry


def history_entry_from_summary(s) -> dict:
    """Build a compact JSON-serializable campaign history dict preserving live regime & electrochem ingredients."""
    tz = s.directional.get("z")
    mspec = s.microstructure_spec or {}
    strain = float(mspec.get("compression_strain", 0.0))
    v_strain = float(mspec.get("virtual_compression_strain", 0.0))
    thickness_ratio = (
        max(0.35, 1.0 - strain)
        if s.source_kind in ("carbon_felt", "porousdiff_gallery")
        else max(0.35, 1.0 - v_strain)
    )
    mech = s.mechanics or {}
    echem = s.electrochem or {}
    morph = s.morphology or {}
    return {
        "run_id": s.run_id,
        "label": s.label,
        "clamping_kpa": float(mspec.get("clamping_kpa", 150.0)),
        "virtual_compression_strain": round(v_strain, 4),
        "porosity": s.porosity,
        "tau_z": tz.tortuosity if tz else None,
        "bruggeman_z": tz.bruggeman_exponent if tz else None,
        "d_eff_z": tz.effective_diffusivity if tz else s.porosity,
        "sigma_z": s.solid_effective_conductivity_z,
        "active_ssa": s.interfacial_area_active_m_inv,
        "thickness_um": round(float(s.shape_zyx[0]) * float(s.voxel_size_um_zyx[0]), 3),
        "thickness_ratio": round(thickness_ratio, 4),
        "r_cell": s.cell_resistance_fom,
        "limiting_c_rate": echem.get("limiting_c_rate"),
        "trapped_fraction": morph.get("trapped_fraction", 0.0),
        "dead_end_fraction": morph.get("dead_end_fraction", 0.0),
        "effective_modulus_gpa": mech.get("effective_modulus_gpa"),
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


def _info_popover(text: str, *, flip: bool = False) -> html.Details:
    """Render a zero-JS `(i)` disclosure popover in plain English."""
    cls = "info-icon info-icon--flip" if flip else "info-icon"
    return html.Details(
        className=cls,
        children=[
            html.Summary("i", title="What does this mean?"),
            html.Div(text, className="info-icon-popover"),
        ],
    )


def _generator_options() -> list[dict]:
    gallery = list_porousdiff_gallery()
    pd_label = (
        f"DLS I12 / PorousDiff · Carbon felt ({len(gallery)} XCT scans)"
        if gallery
        else "PorousDiff AI · Carbon felt"
    )
    return [
        {
            "label": pd_label,
            "value": "porousdiff_gallery",
        },
        {
            "label": "Real 2-phase porous XCT (OpenImpala 100³)",
            "value": "two_phase_xct",
        },
        {
            "label": "Custom carbon felt (synthetic fibres + coating)",
            "value": "carbon_felt",
        },
        {
            "label": "Custom 3-phase battery cathode (NMC + binder)",
            "value": "multiphase_electrode",
        },
        {
            "label": "Sphere packing (synthetic)",
            "value": "overlapping_spheres",
        },
        {
            "label": "Porous foam (random blobs)",
            "value": "gaussian_blobs",
        },
        {
            "label": "Benchmark 8-sphere pack (100³)",
            "value": "benchmark_dataset",
        },
        {
            "label": "Straight-channel test cube (exact check)",
            "value": "open_cube",
        },
    ]


def _physics_options() -> list[dict]:
    sms_ready = has_sms_backend()
    return [
        {
            "label": (
                " Include 3D mechanical stiffness (SMS)"
                if sms_ready
                else " Include 3D mechanical stiffness (install .[sms])"
            ),
            "value": "sms",
            "disabled": not sms_ready,
        },
        {
            "label": " Full 3x3 directional diffusivity tensor",
            "value": "homog",
        },
        {
            "label": " Subvolume size check (REV curve)",
            "value": "rev",
        },
    ]


def build_hub_layout(registry: RunRegistry) -> html.Div:
    """Construct the root Dash layout (`BASE Lab Signature` light + `Warm Darkroom` dark)."""
    init_history = initial_history_records(registry)
    default_id = registry.default_run_id or (init_history[0]["run_id"] if init_history else None)

    operando_map = discover_operando_datasets()
    default_ds_key = (
        "ag_full"
        if "ag_full" in operando_map
        else ("dls_i12_felt" if "dls_i12_felt" in operando_map else next(iter(operando_map.keys())))
    )
    default_ds = operando_map[default_ds_key]
    nz_init, ny_init, nx_init = stack_dimensions(str(default_ds.timepoints[0].path))

    tp_marks = {
        idx: tp.label.split("(")[0].strip()[:12]
        for idx, tp in enumerate(default_ds.timepoints)
    }
    default_tp_idx = min(2, len(default_ds.timepoints) - 1) if default_ds_key == "dls_i12_felt" else (1 if len(default_ds.timepoints) > 1 else 0)
    default_z_idx = nz_init // 2
    default_roi_x = [max(0, nx_init // 2 - 48), min(nx_init, nx_init // 2 + 48)]
    default_roi_y = [max(0, ny_init // 2 - 48), min(ny_init, ny_init // 2 + 48)]
    default_roi_z = [max(0, nz_init // 2 - 24), min(nz_init, nz_init // 2 + 24)]

    chem_options = [
        {"label": c.title, "value": key} for key, c in CHEMISTRIES.items()
    ]

    return html.Div(
        id="hub-theme-wrapper",
        className="hub-shell",
        **{"data-theme": "light"},
        children=[
            # Persistent & session stores
            dcc.Store(id="active-run-id-store", data=default_id),
            dcc.Store(id="run-history-store", data=init_history, storage_type="local"),
            dcc.Store(id="theme-store", data="light", storage_type="local"),
            dcc.Store(id="uploaded-file-store", data=None),
            dcc.Store(id="suggested-kpa-store", data=245.0),
            dcc.Store(id="seed-store", data=42),
            dcc.Download(id="download-bundle"),
            dcc.Download(id="download-bpx"),
            dcc.Download(id="download-campaign"),
            dcc.Download(id="download-volume"),
            # ── Top Navigation Bar (Sibling of jameslehoux.com) ──
            html.Header(
                className="hub-topbar",
                children=[
                    html.A(
                        href="https://jameslehoux.com/",
                        target="_blank",
                        rel="noopener noreferrer",
                        className="hub-brand",
                        title="Visit BASE Laboratory (jameslehoux.com)",
                        children=[
                            html.Img(
                                src="/assets/baselab_logo.png",
                                alt="BASE Laboratory logo",
                            ),
                            html.Span("BASE LAB · OPENIMPALA HUB"),
                        ],
                    ),
                    dcc.Tabs(
                        id="main-view-tabs",
                        value="tab-slices",
                        className="hub-nav-tabs",
                        children=[
                            dcc.Tab(
                                label="Slices",
                                value="tab-slices",
                                className="tab",
                                selected_className="tab--selected",
                            ),
                            dcc.Tab(
                                label="3D View",
                                value="tab-3d",
                                className="tab",
                                selected_className="tab--selected",
                            ),
                            dcc.Tab(
                                label="Compare",
                                value="tab-compare",
                                className="tab",
                                selected_className="tab--selected",
                            ),
                            dcc.Tab(
                                label="Discharge & Pores",
                                value="tab-pores",
                                className="tab",
                                selected_className="tab--selected",
                            ),
                            dcc.Tab(
                                label="Plan Scans",
                                value="tab-plan",
                                className="tab",
                                selected_className="tab--selected",
                            ),
                            dcc.Tab(
                                label="Guide",
                                value="tab-guide",
                                className="tab",
                                selected_className="tab--selected",
                            ),
                        ],
                    ),
                    html.Div(
                        className="hub-header-actions",
                        children=[
                            html.Button(
                                "Dark mode",
                                id="theme-toggle-btn",
                                className="btn btn-subtle",
                                n_clicks=0,
                            ),
                            html.Button(
                                "Download results (.zip)",
                                id="export-bundle-btn",
                                className="btn",
                                n_clicks=0,
                            ),
                        ],
                    ),
                ],
            ),
            # ── Main 3-Column Stage ──
            html.Main(
                id="hub-stage-grid",
                className="hub-stage",
                children=[
                    # ── Left Column: Sample Source + Mode Controls + Technical Details ──
                    html.Aside(
                        className="hub-left-col",
                        children=[
                            html.Div(
                                className="sec-heading",
                                children=[
                                    html.Span("Sample source"),
                                    _info_popover(
                                        "Choose how to get a 3D electrode sample: generate a new synthetic or AI structure (PorousDiff), explore 4D X-ray tomography scans, or upload your own file."
                                    ),
                                ],
                            ),
                            dcc.RadioItems(
                                id="sample-mode-radio",
                                className="source-mode-radio",
                                options=[
                                    {"label": "Generate", "value": "generate"},
                                    {"label": "X-ray Scans", "value": "xray"},
                                    {"label": "Upload", "value": "upload"},
                                ],
                                value="generate",
                            ),
                            # 1. GENERATE PANEL (PorousDiff & Synthetic Generators — Default)
                            html.Div(
                                id="panel-mode-generate",
                                children=[
                                    html.Div(
                                        className="field-group",
                                        children=[
                                            html.Div(
                                                "Generator",
                                                className="field-header",
                                            ),
                                            dcc.Dropdown(
                                                id="source-kind-dropdown",
                                                options=_generator_options(),
                                                value="porousdiff_gallery",
                                                clearable=False,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        id="clamping-field-wrap",
                                        className="field-group",
                                        children=[
                                            html.Div(
                                                className="field-header",
                                                children=[
                                                    html.Span("Compression load"),
                                                    html.Span(
                                                        "238 kPa",
                                                        id="clamping-kpa-readout",
                                                        className="field-readout",
                                                    ),
                                                ],
                                            ),
                                            dcc.Slider(
                                                id="clamping-kpa-slider",
                                                min=0,
                                                max=500,
                                                step=10,
                                                value=238,
                                                marks={0: "0", 68: "68", 238: "238", 451: "451"},
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        id="porosity-field-wrap",
                                        className="field-group",
                                        style={"display": "none"},
                                        children=[
                                            html.Div(
                                                className="field-header",
                                                children=[
                                                    html.Span("Target pore space"),
                                                    html.Span(
                                                        "78%",
                                                        id="porosity-readout",
                                                        className="field-readout",
                                                    ),
                                                ],
                                            ),
                                            dcc.Slider(
                                                id="porosity-slider",
                                                min=0.25,
                                                max=0.88,
                                                step=0.01,
                                                value=0.78,
                                                marks={0.30: "30%", 0.55: "55%", 0.80: "80%"},
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="field-group",
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": "1.25fr 1fr",
                                            "gap": "8px",
                                        },
                                        children=[
                                            html.Div(
                                                children=[
                                                    html.Div("Volume size", className="field-header"),
                                                    dcc.Dropdown(
                                                        id="grid-size-dropdown",
                                                        options=[
                                                            {"label": "48 x 48 x 48", "value": 48},
                                                            {"label": "64 x 64 x 64", "value": 64},
                                                            {"label": "80 x 80 x 80", "value": 80},
                                                            {"label": "96 x 96 x 96", "value": 96},
                                                        ],
                                                        value=64,
                                                        clearable=False,
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                children=[
                                                    html.Div("Pixel scale (µm)", className="field-header"),
                                                    dcc.Input(
                                                        id="voxel-um-input",
                                                        type="number",
                                                        value=3.2,
                                                        min=0.05,
                                                        max=50.0,
                                                        step="any",
                                                        className="num-input",
                                                    ),
                                                ]
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="field-group",
                                        children=[
                                            html.Div("Electrode role (BPX / PyBaMM)", className="field-header"),
                                            dcc.Dropdown(
                                                id="electrode-role-dropdown",
                                                options=[
                                                    {"label": "Negative electrode", "value": "negative"},
                                                    {"label": "Positive cathode", "value": "positive"},
                                                    {"label": "Porous separator", "value": "separator"},
                                                ],
                                                value="negative",
                                                clearable=False,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="field-group",
                                        children=[
                                            dcc.Checklist(
                                                id="physics-options-check",
                                                options=_physics_options(),
                                                value=["sms"] if has_sms_backend() else [],
                                                style={"display": "grid", "gap": "5px", "fontSize": "11.5px"},
                                            ),
                                        ],
                                    ),
                                    html.Button(
                                        "Generate & simulate sample",
                                        id="run-generate-btn",
                                        className="btn btn-primary",
                                        n_clicks=0,
                                    ),
                                    html.Div(
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": "1fr 1fr",
                                            "gap": "8px",
                                            "marginTop": "8px",
                                        },
                                        children=[
                                            html.Button(
                                                "New random seed",
                                                id="new-seed-btn",
                                                className="btn",
                                                n_clicks=0,
                                            ),
                                            html.Button(
                                                "Save 3D volume (.ome.zarr)",
                                                id="save-volume-btn",
                                                className="btn",
                                                n_clicks=0,
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            # 2. X-RAY SCANS PANEL (BatteryExplorer Lazy 4D Loader + Region Drill-Down)
                            html.Div(
                                id="panel-mode-xray",
                                style={"display": "none"},
                                children=[
                                    html.Div(
                                        className="field-group",
                                        children=[
                                            html.Div("Dataset", className="field-header"),
                                            dcc.Dropdown(
                                                id="operando-dataset-dropdown",
                                                options=[
                                                    {"label": ds.label, "value": k}
                                                    for k, ds in operando_map.items()
                                                ],
                                                value=default_ds_key,
                                                clearable=False,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="field-group",
                                        children=[
                                            html.Div("Colour scheme", className="field-header"),
                                            dcc.Dropdown(
                                                id="operando-cmap-dropdown",
                                                options=[
                                                    {"label": "Grayscale", "value": "grayscale"},
                                                    {"label": "Magma", "value": "magma"},
                                                    {"label": "Inferno", "value": "inferno"},
                                                    {"label": "Viridis", "value": "viridis"},
                                                ],
                                                value="grayscale",
                                                clearable=False,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="field-group",
                                        children=[
                                            html.Div(
                                                className="field-header",
                                                children=[
                                                    html.Span("Scan step / load"),
                                                    html.Span(
                                                        default_ds.timepoints[default_tp_idx].label,
                                                        id="operando-time-readout",
                                                        className="field-readout",
                                                    ),
                                                ],
                                            ),
                                            dcc.Slider(
                                                id="operando-time-slider",
                                                min=0,
                                                max=max(0, len(default_ds.timepoints) - 1),
                                                step=1,
                                                value=default_tp_idx,
                                                marks=tp_marks,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="field-group",
                                        children=[
                                            html.Div(
                                                className="field-header",
                                                children=[
                                                    html.Span("Slice depth (Z)"),
                                                    html.Span(
                                                        f"{default_z_idx} of {nz_init - 1}",
                                                        id="operando-z-readout",
                                                        className="field-readout",
                                                    ),
                                                ],
                                            ),
                                            dcc.Slider(
                                                id="operando-z-slider",
                                                min=0,
                                                max=max(0, nz_init - 1),
                                                step=1,
                                                value=default_z_idx,
                                                marks={
                                                    0: "0",
                                                    nz_init // 2: str(nz_init // 2),
                                                    max(0, nz_init - 1): str(max(0, nz_init - 1)),
                                                },
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="sec-heading",
                                        style={"marginTop": "14px"},
                                        children=[
                                            html.Span("Selected region"),
                                            _info_popover(
                                                "Full X-ray stacks contain millions to billions of pixels. Select a sub-box around a crack or pore channel (cyan rectangle on the image) to simulate ion flow and stiffness at full 1x1x1 pixel resolution."
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        id="roi-dims-readout",
                                        className="helper-copy mono",
                                    ),
                                    html.Div(
                                        className="field-group",
                                        children=[
                                            html.Div("Horizontal X (px)", className="field-header"),
                                            dcc.RangeSlider(
                                                id="roi-x-slider",
                                                min=0,
                                                max=nx_init,
                                                step=2,
                                                value=default_roi_x,
                                                marks={0: "0", nx_init: str(nx_init)},
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="field-group",
                                        children=[
                                            html.Div("Vertical Y (px)", className="field-header"),
                                            dcc.RangeSlider(
                                                id="roi-y-slider",
                                                min=0,
                                                max=ny_init,
                                                step=2,
                                                value=default_roi_y,
                                                marks={0: "0", ny_init: str(ny_init)},
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="field-group",
                                        children=[
                                            html.Div("Depth Z (slices)", className="field-header"),
                                            dcc.RangeSlider(
                                                id="roi-z-slider",
                                                min=0,
                                                max=nz_init,
                                                step=2,
                                                value=default_roi_z,
                                                marks={0: "0", nz_init: str(nz_init)},
                                            ),
                                        ],
                                    ),
                                    html.Button(
                                        "Simulate this region",
                                        id="simulate-roi-btn",
                                        className="btn btn-primary",
                                        n_clicks=0,
                                    ),
                                    html.Button(
                                        "Open region in 3D View",
                                        id="open-roi-3d-btn",
                                        className="btn btn-block",
                                        n_clicks=0,
                                    ),
                                ],
                            ),
                            # 3. UPLOAD PANEL (Bring Your Own .tif / .ome.zarr.zip / .npy / .h5)
                            html.Div(
                                id="panel-mode-upload",
                                style={"display": "none"},
                                children=[
                                    dcc.Upload(
                                        id="upload-volume-box",
                                        className="upload-dropzone",
                                        children=html.Div(
                                            [
                                                html.Strong("Drop 3D scan or click to browse"),
                                                html.Br(),
                                                html.Span(
                                                    ".tif, .ome.zarr.zip, .npy, .h5",
                                                    style={"fontSize": "11px"},
                                                ),
                                            ]
                                        ),
                                        multiple=False,
                                    ),
                                    html.Div(
                                        "No file uploaded yet.",
                                        id="upload-status-text",
                                        className="helper-copy",
                                    ),
                                    dcc.Graph(
                                        id="upload-preview-graph",
                                        config={"displayModeBar": False},
                                        style={"height": "175px", "marginBottom": "10px"},
                                    ),
                                    html.Div(
                                        className="field-group",
                                        children=[
                                            html.Div("Threshold mode", className="field-header"),
                                            dcc.Dropdown(
                                                id="upload-seg-mode",
                                                options=[
                                                    {"label": "Automatic (Otsu)", "value": "otsu"},
                                                    {"label": "Manual threshold", "value": "manual"},
                                                    {"label": "Already segmented labels", "value": "labels"},
                                                ],
                                                value="otsu",
                                                clearable=False,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        id="upload-manual-thr-wrap",
                                        className="field-group",
                                        children=[
                                            html.Div("Manual threshold", className="field-header"),
                                            dcc.Slider(
                                                id="upload-manual-thr",
                                                min=0,
                                                max=255,
                                                step=1,
                                                value=128,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="field-group",
                                        children=[
                                            html.Div("Image contrast", className="field-header"),
                                            dcc.RadioItems(
                                                id="upload-contrast-radio",
                                                className="inline-radio",
                                                options=[
                                                    {"label": "Dark = pore", "value": "dark"},
                                                    {"label": "Bright = pore", "value": "bright"},
                                                ],
                                                value="dark",
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="field-group",
                                        children=[
                                            html.Div("Pixel size (µm: Z, Y, X)", className="field-header"),
                                            html.Div(
                                                className="xyz-input-grid",
                                                children=[
                                                    dcc.Input(
                                                        id="upload-dz-um",
                                                        type="number",
                                                        value=1.0,
                                                        min=0.01,
                                                        step="any",
                                                        className="num-input",
                                                    ),
                                                    dcc.Input(
                                                        id="upload-dy-um",
                                                        type="number",
                                                        value=1.0,
                                                        min=0.01,
                                                        step="any",
                                                        className="num-input",
                                                    ),
                                                    dcc.Input(
                                                        id="upload-dx-um",
                                                        type="number",
                                                        value=1.0,
                                                        min=0.01,
                                                        step="any",
                                                        className="num-input",
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="field-group",
                                        children=[
                                            html.Div(
                                                className="field-header",
                                                children=[
                                                    html.Span("Virtual compression"),
                                                    html.Span(
                                                        "0%",
                                                        id="virtual-compress-readout",
                                                        className="field-readout",
                                                    ),
                                                ],
                                            ),
                                            dcc.Slider(
                                                id="virtual-compress-slider",
                                                min=0,
                                                max=40,
                                                step=5,
                                                value=0,
                                                marks={0: "0%", 20: "20%", 40: "40%"},
                                            ),
                                        ],
                                    ),
                                    html.Button(
                                        "Simulate uploaded scan",
                                        id="run-upload-btn",
                                        className="btn btn-primary",
                                        n_clicks=0,
                                    ),
                                ],
                            ),
                            # 4. CONTEXTUAL 3D VIEW CONTROLS (shown when 3D View tab is active)
                            html.Div(
                                id="panel-3d-controls",
                                style={"display": "none", "marginTop": "16px"},
                                children=[
                                    html.Div(
                                        className="sec-heading",
                                        children=[
                                            html.Span("3D View settings"),
                                            _info_popover(
                                                "Adjust surface brightness threshold, transparency, and cutaway planes. Outer rim masking hides bright X-ray edge artefacts so interior cracks are visible."
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="field-group",
                                        children=[
                                            html.Div("Volume to show in 3D", className="field-header"),
                                            dcc.Dropdown(
                                                id="vol-source-dropdown",
                                                options=[
                                                    {"label": "Active simulated sample", "value": "sim"},
                                                    {"label": "Full X-ray stack (overview)", "value": "xray_full"},
                                                    {"label": "Selected X-ray region (1x1x1)", "value": "xray_roi"},
                                                ],
                                                value="sim",
                                                clearable=False,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="field-group",
                                        children=[
                                            html.Div(
                                                className="field-header",
                                                children=[
                                                    html.Span("Surface threshold"),
                                                    html.Span("128", id="vol-threshold-readout", className="field-readout"),
                                                ],
                                            ),
                                            dcc.Slider(
                                                id="vol-threshold-slider",
                                                min=16,
                                                max=240,
                                                step=4,
                                                value=128,
                                                marks={32: "32", 128: "128", 224: "224"},
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="field-group",
                                        children=[
                                            html.Div("Surface opacity", className="field-header"),
                                            dcc.Slider(
                                                id="vol-opacity-slider",
                                                min=0.15,
                                                max=1.0,
                                                step=0.05,
                                                value=0.95,
                                                marks={0.2: "0.2", 0.6: "0.6", 1.0: "1.0"},
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="field-group",
                                        children=[
                                            html.Div("Hide bright outer rim (r/R)", className="field-header"),
                                            dcc.Slider(
                                                id="vol-rim-slider",
                                                min=0.40,
                                                max=1.00,
                                                step=0.02,
                                                value=1.00,
                                                marks={0.5: "0.5", 0.8: "0.8", 1.0: "Off"},
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="field-group",
                                        children=[
                                            html.Div("Cutaway X / Y / Z (%)", className="field-header"),
                                            dcc.RangeSlider(
                                                id="vol-clip-x",
                                                min=0,
                                                max=100,
                                                step=5,
                                                value=[0, 100],
                                                marks={0: "X0", 100: "X1"},
                                            ),
                                            dcc.RangeSlider(
                                                id="vol-clip-y",
                                                min=0,
                                                max=100,
                                                step=5,
                                                value=[0, 100],
                                                marks={0: "Y0", 100: "Y1"},
                                            ),
                                            dcc.RangeSlider(
                                                id="vol-clip-z",
                                                min=0,
                                                max=100,
                                                step=5,
                                                value=[0, 100],
                                                marks={0: "Z0", 100: "Z1"},
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            # 5. QUIET TECHNICAL DETAILS DISCLOSURE (Solver telemetry + BPX JSON)
                            html.Details(
                                className="tech-details",
                                children=[
                                    html.Summary("Technical details"),
                                    html.Div(
                                        id="tech-telemetry-line",
                                        style={"marginTop": "6px", "lineHeight": "1.5"},
                                    ),
                                    html.Div(
                                        style={"marginTop": "8px"},
                                        children=[
                                            html.Button(
                                                "Download BPX parameters (.json)",
                                                id="export-bpx-btn",
                                                className="btn",
                                                style={"padding": "5px 9px", "fontSize": "11px"},
                                                n_clicks=0,
                                            )
                                        ],
                                    ),
                                    html.Pre(id="pybamm-bpx-preview"),
                                ],
                            ),
                        ],
                    ),
                    # ── Center Column: Viewports & Full-Width Analytical Views ──
                    html.Section(
                        className="hub-center-col",
                        children=[
                            # VIEW 1: SLICES
                            html.Div(
                                id="view-slices",
                                className="micrograph-stage-wrap",
                                children=[
                                    html.Div(
                                        id="slice-toolbar-row",
                                        className="slice-toolbar",
                                        children=[
                                            dcc.RadioItems(
                                                id="slice-display-mode",
                                                className="inline-radio",
                                                options=[
                                                    {"label": "Simulated sample", "value": "sim"},
                                                    {"label": "Raw X-ray scan", "value": "xray"},
                                                ],
                                                value="sim",
                                            ),
                                            dcc.RadioItems(
                                                id="slice-plane-radio",
                                                className="inline-radio",
                                                options=[
                                                    {"label": "XZ (through thickness)", "value": "xz"},
                                                    {"label": "XY (in-plane)", "value": "xy"},
                                                    {"label": "YZ", "value": "yz"},
                                                ],
                                                value="xz",
                                            ),
                                            dcc.RadioItems(
                                                id="slice-layer-radio",
                                                className="inline-radio",
                                                options=[
                                                    {"label": "Ion current", "value": "flux"},
                                                    {"label": "Voltage", "value": "potential"},
                                                    {"label": "Phases", "value": "phase"},
                                                    {"label": "Stress", "value": "vm_stress"},
                                                ],
                                                value="flux",
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="micrograph-frame",
                                        children=[
                                            html.Img(
                                                id="primary-slice-img",
                                                className="lazy-slice-img",
                                                alt="2D cross-section",
                                            )
                                        ],
                                    ),
                                    html.Div(
                                        className="stage-caption",
                                        children=[
                                            html.Span(id="primary-caption-left"),
                                            html.Span(id="primary-caption-right", className="mono"),
                                        ],
                                    ),
                                    html.Div(
                                        id="sim-slice-scrub-row",
                                        style={
                                            "width": "100%",
                                            "maxWidth": "680px",
                                            "marginTop": "12px",
                                            "display": "grid",
                                            "gridTemplateColumns": "1fr auto",
                                            "gap": "12px",
                                            "alignItems": "center",
                                        },
                                        children=[
                                            dcc.Slider(
                                                id="run-slice-slider",
                                                min=0,
                                                max=47,
                                                step=1,
                                                value=24,
                                            ),
                                            html.Button(
                                                "Jump to tightest bottleneck",
                                                id="jump-roi-btn",
                                                className="btn",
                                                n_clicks=0,
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            # VIEW 2: 3D VIEW
                            html.Div(
                                id="view-3d",
                                style={"display": "none"},
                                className="micrograph-stage-wrap",
                                children=[
                                    html.Div(
                                        className="micrograph-frame",
                                        children=[
                                            dcc.Graph(
                                                id="volume-3d-graph",
                                                style={"height": "540px"},
                                            )
                                        ],
                                    ),
                                    html.Div(
                                        className="stage-caption",
                                        children=[
                                            html.Span(id="vol-3d-caption-left"),
                                            html.Span(id="vol-3d-caption-right", className="mono"),
                                        ],
                                    ),
                                ],
                            ),
                            # VIEW 3: COMPARE
                            html.Div(
                                id="view-compare",
                                style={"display": "none"},
                                children=[
                                    html.Div(
                                        className="sec-heading",
                                        children=[
                                            html.Span("Side-by-side comparison"),
                                            _info_popover(
                                                "Compare two X-ray cycling timepoints (with shared brightness histogram axes and lockstep Z slicing) or compare two simulated samples side-by-side.",
                                                flip=True,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        style={
                                            "display": "flex",
                                            "flexWrap": "wrap",
                                            "justifyContent": "space-between",
                                            "alignItems": "center",
                                            "gap": "14px",
                                            "marginBottom": "14px",
                                        },
                                        children=[
                                            dcc.RadioItems(
                                                id="cmp-mode-radio",
                                                className="inline-radio",
                                                options=[
                                                    {"label": "X-ray cycling timepoints", "value": "xray"},
                                                    {"label": "Simulated samples", "value": "runs"},
                                                ],
                                                value="xray",
                                            ),
                                            dcc.Checklist(
                                                id="cmp-link-z-toggle",
                                                options=[{"label": " Link slice sliders", "value": "link"}],
                                                value=["link"],
                                                style={"fontSize": "12px", "fontWeight": "600"},
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="compare-grid",
                                        children=[
                                            html.Div(
                                                children=[
                                                    dcc.Dropdown(
                                                        id="cmp-left-select",
                                                        clearable=False,
                                                    ),
                                                    html.Div(
                                                        style={"margin": "8px 0"},
                                                        children=[
                                                            dcc.Slider(
                                                                id="cmp-left-z",
                                                                min=0,
                                                                max=max(0, nz_init - 1),
                                                                step=1,
                                                                value=default_z_idx,
                                                                marks={
                                                                    0: "0",
                                                                    nz_init // 2: str(nz_init // 2),
                                                                    max(0, nz_init - 1): str(max(0, nz_init - 1)),
                                                                },
                                                            )
                                                        ],
                                                    ),
                                                    html.Div(
                                                        className="micrograph-frame",
                                                        children=[
                                                            html.Img(
                                                                id="cmp-left-img",
                                                                className="lazy-slice-img",
                                                                alt="Left comparison slice",
                                                            )
                                                        ],
                                                    ),
                                                    html.Div(
                                                        id="cmp-left-caption",
                                                        className="helper-copy mono",
                                                    ),
                                                    dcc.Graph(
                                                        id="cmp-left-hist",
                                                        config={"displayModeBar": False},
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                children=[
                                                    dcc.Dropdown(
                                                        id="cmp-right-select",
                                                        clearable=False,
                                                    ),
                                                    html.Div(
                                                        style={"margin": "8px 0"},
                                                        children=[
                                                            dcc.Slider(
                                                                id="cmp-right-z",
                                                                min=0,
                                                                max=max(0, nz_init - 1),
                                                                step=1,
                                                                value=default_z_idx,
                                                                marks={
                                                                    0: "0",
                                                                    nz_init // 2: str(nz_init // 2),
                                                                    max(0, nz_init - 1): str(max(0, nz_init - 1)),
                                                                },
                                                            )
                                                        ],
                                                    ),
                                                    html.Div(
                                                        className="micrograph-frame",
                                                        children=[
                                                            html.Img(
                                                                id="cmp-right-img",
                                                                className="lazy-slice-img",
                                                                alt="Right comparison slice",
                                                            )
                                                        ],
                                                    ),
                                                    html.Div(
                                                        id="cmp-right-caption",
                                                        className="helper-copy mono",
                                                    ),
                                                    dcc.Graph(
                                                        id="cmp-right-hist",
                                                        config={"displayModeBar": False},
                                                    ),
                                                ]
                                            ),
                                        ],
                                    ),
                                    html.Div(id="cmp-delta-summary", style={"marginTop": "14px"}),
                                ],
                            ),
                            # VIEW 4: DISCHARGE & PORES
                            html.Div(
                                id="view-pores",
                                style={"display": "none"},
                                children=[
                                    html.Div(
                                        className="sec-heading",
                                        children=[
                                            html.Span("Simulated battery discharge & voltage losses"),
                                            html.Div(
                                                style={"width": "260px"},
                                                children=[
                                                    dcc.Dropdown(
                                                        id="chemistry-dropdown",
                                                        options=chem_options,
                                                        value="vanadium_flow_felt",
                                                        clearable=False,
                                                    )
                                                ],
                                            ),
                                        ],
                                    ),
                                    dcc.Graph(id="electrochem-graph"),
                                    html.Div(
                                        className="sec-heading",
                                        style={"marginTop": "18px"},
                                        children=[
                                            html.Span("Pore sizes, connectivity & directional pathways"),
                                            _info_popover(
                                                "Percolating pores connect both faces of the electrode and carry steady current. Dead-end pores connect to one side only; trapped pores are sealed inside solid.",
                                                flip=True,
                                            ),
                                        ],
                                    ),
                                    dcc.Graph(id="morphology-graph"),
                                    dcc.Graph(id="directional-profile-graph"),
                                ],
                            ),
                            # VIEW 5: PLAN SCANS
                            html.Div(
                                id="view-plan",
                                style={"display": "none"},
                                children=[
                                    html.Div(
                                        className="sec-heading",
                                        children=[
                                            html.Span("Plan next compression test"),
                                            _info_popover(
                                                "Compressing a porous electrode improves solid electrical contact, but squeezing too hard chokes off ion pathways in the pores. A Gaussian Process model learns this trade-off and suggests the most informative next load.",
                                                flip=True,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        id="gp-recommendation-banner",
                                        style={"marginBottom": "12px", "fontSize": "13px"},
                                    ),
                                    html.Div(
                                        style={
                                            "display": "flex",
                                            "flexWrap": "wrap",
                                            "gap": "10px",
                                            "marginBottom": "16px",
                                        },
                                        children=[
                                            html.Button(
                                                "Run recommended load",
                                                id="run-suggested-load-btn",
                                                className="btn btn-primary",
                                                style={"width": "auto"},
                                                n_clicks=0,
                                            ),
                                            html.Button(
                                                "Run 5-load compression sweep",
                                                id="run-5pt-sweep-btn",
                                                className="btn",
                                                n_clicks=0,
                                            ),
                                            html.Button(
                                                "Save campaign (.json)",
                                                id="save-campaign-btn",
                                                className="btn",
                                                n_clicks=0,
                                            ),
                                            dcc.Upload(
                                                id="load-campaign-upload",
                                                children=html.Button(
                                                    "Load campaign (.json)",
                                                    className="btn",
                                                    type="button",
                                                ),
                                                multiple=False,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": "repeat(3, minmax(0, 1fr))",
                                            "gap": "16px",
                                            "marginBottom": "12px",
                                        },
                                        children=[
                                            html.Div(
                                                children=[
                                                    html.Div("Ion pathway weight", className="field-header"),
                                                    dcc.Slider(
                                                        id="gp-w-ion",
                                                        min=0.05,
                                                        max=1.20,
                                                        step=0.05,
                                                        value=0.58,
                                                        marks={0.1: "0.1", 0.6: "0.6", 1.2: "1.2"},
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                children=[
                                                    html.Div("Electrical contact weight", className="field-header"),
                                                    dcc.Slider(
                                                        id="gp-w-elec",
                                                        min=0.05,
                                                        max=1.20,
                                                        step=0.05,
                                                        value=0.27,
                                                        marks={0.1: "0.1", 0.6: "0.6", 1.2: "1.2"},
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                children=[
                                                    html.Div("Surface reaction weight", className="field-header"),
                                                    dcc.Slider(
                                                        id="gp-w-kin",
                                                        min=0.05,
                                                        max=0.80,
                                                        step=0.05,
                                                        value=0.15,
                                                        marks={0.1: "0.1", 0.4: "0.4", 0.8: "0.8"},
                                                    ),
                                                ]
                                            ),
                                        ],
                                    ),
                                    dcc.Graph(id="gp-steering-graph"),
                                ],
                            ),
                            # VIEW 6: GUIDE
                            html.Div(
                                id="view-guide",
                                style={"display": "none"},
                                children=[
                                    html.Div(
                                        className="sec-heading",
                                        children=[
                                            html.Span("Interactive guide & verification exercises"),
                                        ],
                                    ),
                                    html.P(
                                        "New to 3D battery microstructure modelling? Work through the four short exercises below. Each exercise runs a live 3D simulation so you can verify how pore space, pathway tortuosity, and compression shape battery performance.",
                                        style={"maxWidth": "760px", "marginBottom": "16px"},
                                    ),
                                    html.Div(
                                        style={
                                            "display": "flex",
                                            "flexWrap": "wrap",
                                            "gap": "10px",
                                            "marginBottom": "18px",
                                        },
                                        children=[
                                            html.Button(
                                                "1. Run straight-channel cube",
                                                id="preset-btn-cube",
                                                className="btn",
                                                n_clicks=0,
                                            ),
                                            html.Button(
                                                "2. Run 8-sphere benchmark",
                                                id="preset-btn-spheres",
                                                className="btn",
                                                n_clicks=0,
                                            ),
                                            html.Button(
                                                "3. Run 3-phase cathode",
                                                id="preset-btn-cathode",
                                                className="btn",
                                                n_clicks=0,
                                            ),
                                            html.Button(
                                                "4. Run 5-load felt sweep",
                                                id="guide-sweep-btn",
                                                className="btn",
                                                n_clicks=0,
                                            ),
                                        ],
                                    ),
                                    html.Div(id="guide-cards-container", className="guide-grid"),
                                ],
                            ),
                        ],
                    ),
                    # ── Right Column: Plain-English Simulation Results + Context Charts ──
                    html.Aside(
                        id="hub-right-column",
                        className="hub-right-col",
                        children=[
                            html.Div(
                                className="sec-heading",
                                children=[
                                    html.Span("Simulation results"),
                                    _info_popover(
                                        "Key transport and mechanical properties solved directly on the 3D voxel grid. Tortuosity near 1.0 means straight, open channels; higher values mean ions must take longer detours around solid particles.",
                                        flip=True,
                                    ),
                                ],
                            ),
                            html.Div(id="active-sample-subtitle", className="helper-copy"),
                            html.Table(id="results-summary-table", className="results-table"),
                            # Context chart A: Simulated discharge (0.5C - 4C)
                            html.Div(
                                id="right-panel-sim-chart",
                                children=[
                                    html.Div(
                                        className="sec-heading",
                                        children=[
                                            html.Span("Simulated discharge (0.5C – 4C)"),
                                            _info_popover(
                                                "Predicted cell voltage versus delivered capacity at slow (0.5C) to fast (4C) discharge rates using your sample's solved ion diffusivity and electrical wiring.",
                                                flip=True,
                                            ),
                                        ],
                                    ),
                                    dcc.Graph(
                                        id="compact-discharge-graph",
                                        config={"displayModeBar": False},
                                    ),
                                ],
                            ),
                            # Context chart B: Operando cycling / compression load + structural traces
                            html.Div(
                                id="right-panel-xray-charts",
                                style={"display": "none"},
                                children=[
                                    html.Div(
                                        className="sec-heading",
                                        children=[
                                            html.Span(
                                                "In-situ clamping load (DLS I12)",
                                                id="xray-echem-title",
                                            ),
                                            _info_popover(
                                                "Displays clamping pressure for mechanical compression scans, cell voltage for operando cycling scans, or live slice brightness distribution for single scans.",
                                                flip=True,
                                            ),
                                        ],
                                    ),
                                    dcc.Graph(
                                        id="operando-echem-graph",
                                        config={"displayModeBar": False},
                                    ),
                                    html.Div(
                                        id="xray-struct-section",
                                        children=[
                                            html.Div(
                                                className="sec-heading",
                                                style={"marginTop": "10px"},
                                                children=[
                                                    html.Span("Structural changes during cycling"),
                                                    _info_popover(
                                                        "Top: brightness at the plating interface. Middle: outer rim vs centre check (flat = stable beam). Bottom: volume fraction of dark voids and plated lithium in the active layer.",
                                                        flip=True,
                                                    ),
                                                ],
                                            ),
                                            dcc.Graph(
                                                id="operando-structural-graph",
                                                config={"displayModeBar": False},
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            # Context chart C: 3D Brightness Histogram (when on 3D View)
                            html.Div(
                                id="right-panel-3d-hist",
                                style={"display": "none"},
                                children=[
                                    html.Div(
                                        className="sec-heading",
                                        children=[html.Span("3D brightness distribution")],
                                    ),
                                    dcc.Graph(
                                        id="vol-3d-hist-graph",
                                        config={"displayModeBar": False},
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
