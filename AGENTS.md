# Agent & Contributor Guidelines (`OpenImpala-Hub`)

Read [ECOSYSTEM.md](https://github.com/BASE-Laboratory/basecoat/blob/main/ECOSYSTEM.md) before adding a module, dependency, file reader, or dataset constant. The canonical copy of `ECOSYSTEM.md` lives in `BASE-Laboratory/basecoat`.

## Ecosystem Layer

- **Repository:** `BASE-Laboratory/OpenImpala-Hub`
- **Licence:** `BSD-3-Clause`
- **Layer:** **Layer 4 — Workflows, benchmarks, and services**
- **Allowed ecosystem dependencies:** `basecoat` (Layer 1), `PorousDiff` (Layer 2, optional checkpoint/gallery extra), and `openimpala` (Layer 3).
- **Reverse-dependency rule:** No Layer 1, Layer 2, or Layer 3 package may import `openimpala_hub`.

## Ownership Boundaries

- **Owns:** Interactive Plotly Dash web service (`openimpala_hub.app`), JSON/CLI run contracts (`openimpala_hub.contracts`, `openimpala_hub.cli`), procedural fibrous carbon-felt & benchmark microstructure staging (`openimpala_hub.engine.generators`), autonomous experiment steering (`openimpala_hub.engine.steering`: 3D diffusive flux bottleneck ROI selection & 1D Gaussian Process setpoint planning), reproducible `.zip` NIA report bundle export (`openimpala_hub.engine.bundle`), and container/serverless deployment (`Dockerfile`, `docker-compose.yml`, `deploy/modal_app.py`).
- **Delegates to `basecoat` (`v0.3`):** Calibrated `Volume`, `VolumeSet`, `AxisSpec`, and `LazyVolume` containers, out-of-core region reading (`open_volume`), OME-Zarr v0.4 multiscales (`read_ome_zarr`, `write_ome_zarr`), and layer import enforcement (`python -m basecoat.layers`).
- **Delegates to `openimpala` (`>=5.0,<6.0`):** Steady-state finite-volume diffusive transport (`tortuosity`, `tortuosity_all`, `effective_diffusivity`), geometric stereology (`volume_fraction`, `percolation_check`, `specific_surface_area`, `through_thickness_profile`), synthetic baseline generators (`openimpala.synthetic`), and PyBaMM / BPX parameterisation (`to_pybamm_parameters`, `export_bpx_json`).
