"""Headless simulation, synthesis, autonomous steering, and export engine for OpenImpala-Hub."""

from __future__ import annotations

from .bridge import (
     enforce_solve_grid_cap,
    extract_phase_volume,
    resolve_voxel_size_metres,
)
from .bundle import build_report_bundle_bytes, write_report_bundle
from .generators import (
    generate_carbon_felt,
    list_porousdiff_gallery,
    load_or_generate_microstructure,
)
from .solver import ExecutedRun, run_transport_pipeline
from .steering import (
    compute_cell_resistance,
    find_flux_bottleneck_rois,
    suggest_next_setpoint_gp,
)

__all__ = [
    "ExecutedRun",
    "build_report_bundle_bytes",
    "compute_cell_resistance",
    "enforce_solve_grid_cap",
    "extract_phase_volume",
    "find_flux_bottleneck_rois",
    "generate_carbon_felt",
    "list_porousdiff_gallery",
    "load_or_generate_microstructure",
    "resolve_voxel_size_metres",
    "run_transport_pipeline",
    "suggest_next_setpoint_gp",
    "write_report_bundle",
]
