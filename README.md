---
title: The OpenImpala Hub
emoji: 🔋
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
license: bsd-3-clause
---

# The OpenImpala Hub (`v0.4.2`)

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Layer 4: Service](https://img.shields.io/badge/BASE_Lab-Layer_4_Service-0d9488)](AGENTS.md)
[![OpenImpala](https://img.shields.io/badge/Solver-OpenImpala_v5.x-2563eb)](https://github.com/BASE-Laboratory/OpenImpala)
[![SMS Mechanics](https://img.shields.io/badge/Mechanics-SMS_Fortran_3D-7c3aed)](https://github.com/BASE-Laboratory/structural_mechanics_solver)
[![Live Browser App](https://img.shields.io/badge/Live_Demo-GitHub_Pages-0035F5)](https://base-laboratory.github.io/OpenImpala-Hub/)

**The OpenImpala Hub** is an interactive browser workbench for **3D electrode generation (`PorousDiff` & synthetic models)**, **out-of-core 4D X-ray tomography exploration (`BatteryExplorer`)**, **AMReX/HYPRE transport simulation (`OpenImpala`)**, **3D finite-volume elasticity (`SMS`)**, **simulated battery discharge (`0.5C–4C`)**, and **Bayesian compression planning**.

- **Always-On Browser Workbench (£0/mo GitHub Pages + OME-Zarr HTTP Chunk Streaming):** [https://base-laboratory.github.io/OpenImpala-Hub/](https://base-laboratory.github.io/OpenImpala-Hub/)

Developed by the **[BASE Laboratory](https://jameslehoux.com/)** at the University of Greenwich ([GitHub](https://github.com/BASE-Laboratory)).

- **Maintainer:** Dr. James Le Houx (`j.lehoux@greenwich.ac.uk`)
- **Contributors:** James Le Houx (PI, `OpenImpala` / `basecoat` / `SMS` multiphysics engine & design system), Ali Humza Butt ([`BatteryExplorer`](https://github.com/BASE-Laboratory/BatteryExplorer) lazy TIFF/Zarr streaming, Dask slab pooling & 4D operando degradation metrics), Asif Ali ([`PorousDiff`](https://github.com/BASE-Laboratory/PorousDiff) 3D load-conditioned diffusion synthesis & Diamond `I12` carbon-felt calibration)

---

## Key Capabilities (`v0.4.1`)

1. **Three Sample Sources Front-and-Centre (`Generate` · `X-ray Scans` · `Upload`)**
   - **`Generate` (`DLS I12 / PorousDiff` & Custom 3D Models):** Boots directly onto real **Diamond Light Source `I12` compressed virgin carbon-felt synchrotron scans (`68`, `238`, `451 kPa`)** alongside `PorousDiff` galleries, real `100³` two-phase porous micro-CT (`SampleData_2Phase_stack_3d_1bit.tif`), custom fibrous felt with catalyst shells, 3-phase battery cathodes (`NMC + binder`), sphere packings, random porous foams, and exact analytical verification cubes — with 1-click `.ome.zarr.zip` volume export.
   - **`X-ray Scans` (`BatteryExplorer` Out-of-Core 4D Engine + Bundled Synchrotron Data):** Ships with `16.7 MB` of RFC 1950 `zlib` multiscale OME-Zarr v0.4 stores shared zero-copy between Python (`basecoat`) and browser `DecompressionStream('deflate')`:
     - **Diamond Light Source `I12` In-Situ Virgin Carbon Felt (`68`, `238`, `451 kPa` @ `1.6 µm/px`, `128 × 192 × 192`)**
     - **Diamond Light Source `I13-2` Zeolite 13X Capillary + `500 µm` Ruby Fiducial (`3.25 µm/px`, `48 × 400 × 400`, Zenodo [`10.5281/zenodo.14245751`](https://doi.org/10.5281/zenodo.14245751), Green et al., *Sci. Data* 12, 329 (2025))**
     - **Segmented Dendritic / Porous Electrode Micro-CT (`binarised_nlm25`, `2.0 µm/px`, `128 × 256 × 256`)**
     - Supports full-resolution external `ag_cell` / `cu_cell` operando `.tif` / `.ome.zarr` stacks via `$OPENIMPALA_HUB_OPERANDO_DIR`, thread-safe lazy slice streaming, `dask.delayed` 3D slab pooling, radial Hough-circle rim masking (`r/R`), lockstep-Z comparison, and **native `1x1x1` (`step=1`) selected-region physics simulation**.
   - **`Upload` (Bring Your Own Scan):** UUID-isolated `.tif`, `.ome.zarr.zip`, `.npy`, and `.h5` ingestion with live Otsu/manual threshold preview, anisotropic `(dz, dy, dx)` µm calibration, and solid-mass-conserving virtual Z-compression.
2. **Coupled Layer-3 Transport (`OpenImpala`) & 3D Mechanics (`SMS`)**
   - Solves 3-axis pathway tortuosity $(\tau_x, \tau_y, \tau_z)$, effective ion diffusivity, solid electrical wiring, $3\times 3$ periodic homogenisation tensors, subvolume REV convergence, 6-connected **Percolating / Dead-End / Trapped** pore partitioning, and optional 3D Young's modulus + von Mises stress fields (`SMS`).
3. **Simulated Battery Discharge (`0.5C–4C`), Bayesian Scan Planning & Plain-English Design**
   - Styled directly alongside [`jameslehoux.com`](https://jameslehoux.com/) (`BASE Lab Signature` `#FCD0B1` / `#2F3E46` / `#0035F5` + persistent `Warm Darkroom` `#222222` dark mode in pure Vanilla CSS), pairing plain-English explanations and `(i)` disclosure popovers with exact PyBaMM / Faraday BPX exports and reproducible `.zip` report bundles.

---

## Run Locally in Under 10 Minutes

### Option A — Docker Compose

```bash
git clone https://github.com/BASE-Laboratory/OpenImpala-Hub.git
cd OpenImpala-Hub
docker compose up --build
```

### Option B — Python Virtual Environment (`Linux x86_64` C++/MPI + Optional Fortran `SMS`)

```bash
git clone https://github.com/BASE-Laboratory/OpenImpala-Hub.git
cd OpenImpala-Hub
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[mpi,sms,dev]"  # omit ,sms if gfortran or private SMS access is unavailable

# Verify Layer 4 boundaries & analytical closed-form sanity gate (tau = 1.0000)
python -m basecoat.layers .
openimpala-hub verify

# Launch interactive web workbench on port 7860
openimpala-hub serve --host 0.0.0.0 --port 7860
```

---

## CLI & JSON REST API Contract

### Command-Line Interface

```bash
openimpala-hub --help
openimpala-hub presets
openimpala-hub verify
openimpala-hub run --preset carbon_felt_compression --clamping-kpa 180 --mechanics --chemistry vrfb_carbon_felt --output-bundle report_180kpa.zip
openimpala-hub run --preset composite_cathode_3phase --electrode-role positive --virtual-compress-strain 0.20 --chemistry nmc811_li_ion --json
```

### HTTP JSON Endpoints

- `GET /healthz` — Container health, `hub_version` (`0.4.0`), `openimpala` / `basecoat` versions, and active solver backend.
- `GET /api/v1/presets` — Machine-readable preset catalogue.
- `POST /api/v1/upload` — Stage `.tif`, `.npy`, `.h5`, or `.ome.zarr.zip` into UUID-isolated storage with zip-slip validation and return pre-solve histogram & calibration metadata.
- `POST /api/v1/solve` — Execute a preset or custom `{"microstructure": {...}, "solve": {...}}` payload (enforcing `HUB_MAX_SOLVE_SIDE` and staged-upload verification) and return `SolveRunSummary` JSON.
- `GET /api/v1/bundle/<run_id>.zip` — Stream the reproducible `.zip` report bundle (`manifest.json`, `metrics.json`, `bpx_parameters.json`, `reproduce.py`, publication PNGs, and `microstructure.ome.zarr`).

---

## Cloud, GitHub Pages & Serverless GPU Deployment

- **Always-On Static Browser Workbench (£0/month GitHub Pages):** Built via [`scripts/build_pages_zarr_demo.py`](scripts/build_pages_zarr_demo.py) and deployed from [`docs_site/`](docs_site/) by [`.github/workflows/pages.yml`](.github/workflows/pages.yml) to `https://base-laboratory.github.io/OpenImpala-Hub/`.
- **Container PaaS / VPS (`Dockerfile`):** Listens on `PORT=7860` as non-root `uid=1000` with `/healthz` healthchecks.
- **Zero-Idle-Cost Serverless + Per-Second L4 GPU Burst (`deploy/modal_app.py`):**
  ```bash
  pip install -e ".[modal]"
  python deploy/modal_app.py --check
  modal deploy deploy/modal_app.py
  ```

---

## Ecosystem Governance & Licence

`OpenImpala-Hub` is a **Layer 4 Service** in the BASE Laboratory 4-layer architecture ([`ECOSYSTEM.md`](https://github.com/BASE-Laboratory/basecoat/blob/main/ECOSYSTEM.md)) and is released under the **BSD 3-Clause License** ([`LICENSE`](LICENSE)).
