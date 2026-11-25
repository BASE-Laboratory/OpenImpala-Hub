---
title: The OpenImpala Hub
emoji: 🔋
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
license: bsd-3-clause
---

# The OpenImpala Hub

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Layer 4: Service](https://img.shields.io/badge/BASE_Lab-Layer_4_Service-0d9488)](AGENTS.md)
[![OpenImpala](https://img.shields.io/badge/Solver-OpenImpala_v5.x-2563eb)](https://github.com/BASE-Laboratory/OpenImpala)

**The OpenImpala Hub** is an integrated, browser-accessible computational platform coupling **3D microstructure synthesis**, **AMReX/HYPRE transport physics simulations (`OpenImpala`)**, **out-of-core volumetric visualisation**, and **autonomous experiment steering** in a zero-installation workflow.

Developed by the **[BASE Laboratory](https://github.com/BASE-Laboratory)** at the University of Greenwich (CSEG ECR Seed Award prototype supporting the EPSRC New Investigator Award & *Heuristic Operando* digital-twin programme).

- **Maintainer:** Dr. James Le Houx (`j.lehoux@greenwich.ac.uk`)
- **Contributors:** James Le Houx (PI, solver & digital-twin engine), Ali Humza Butt (volumetric web dashboard research strand), Asif Ali (3D diffusion microstructure synthesis research strand)

---

## Three Unified Pillars + Autonomous Steering

1. **3D Microstructure Synthesis & Ingestion (`openimpala_hub.engine.generators`)**
   - **Procedural Anisotropic Generators:** Non-woven fibrous **Carbon Felt** with mechanical Z-compression (`kPa` $\rightarrow$ compressive strain $\epsilon_z$) and surface-supported catalyst/binder shells, plus `openimpala.synthetic` (`gaussian_blobs`, `overlapping_spheres`, `multiphase_electrode`) and analytical verification domains.
   - **Drop-In `PorousDiff` 3D Diffusion Gallery:** Automatically discovers calibrated `.ome.zarr` conditional 3D diffusion realisations exported from Isambard-AI GH200 (`data/porousdiff/` or `$OPENIMPALA_HUB_POROUSDIFF_DIR`).
   - **Out-of-Core Experimental XCT / FIB-SEM (`v0.2`):** Region-lazy `.ome.zarr.zip`, `.tif`, `.npy`, and `.h5` ingestion via [`basecoat`](https://github.com/BASE-Laboratory/basecoat) (`v0.3`) with pre-solve greyscale histogram & live Otsu/manual/polarity (`Dark = Pore` vs `Bright = Pore`) segmentation preview, UUID-isolated upload staging, and calibrated anisotropic `(dz, dy, dx)` SI micrometre spacing.
2. **High-Performance Transport Physics (`openimpala_hub.engine.solver`)**
   - Powered by [`OpenImpala`](https://github.com/BASE-Laboratory/OpenImpala) (`pybind11` C++/MPI matrix-free `AMReX MLMG` and `HYPRE` solvers, with automatic pure-Python `SciPy` fallback).
   - Evaluates 3-axis tortuosity $(\tau_x, \tau_y, \tau_z)$, effective diffusivity, composite multiphase conductances, $3\times 3$ periodic homogenisation tensors, subvolume **Representative Elementary Volume (REV)** convergence curves (`openimpala.rev_study`), unbiased $2/3$ Cauchy–Crofton specific surface area, solved 3D potential $\phi(z,y,x)$, cell-centred diffusive flux crowding $|\mathbf{J}/J_0|$, and role-aware (`negative`, `positive`, `separator`) **PyBaMM / Faraday BPX** continuum parameter sets.
3. **Volumetric Browser Dashboard & Reproducible Export (`openimpala_hub.app`)**
   - **2D Slices & Flux Crowding:** Interactive `XZ` / `XY` / `YZ` planes overlaying phase labels, steady-state potential $\phi$, and local diffusive flux crowding $|\mathbf{J}/J_0|$ with plane-aware bounds for rectangular tomograms.
   - **3D Cutaway Viewer:** Tab-gated marching-cubes phase surfaces with interactive `X`/`Z` clip-boxes and 3D synchrotron Zoom-ROI wireframes.
   - **Autonomous Planner (Digital Twin):**
     - *Spatial Beamline Trigger:* Ranks non-overlapping 3D sub-volumes (`ZoomROI`) with peak current crowding and throat constriction for targeted synchrotron zoom XCT / XRD-CT, with 1-click slice jump.
     - *Bayesian Setpoint Planner:* Fits a Matérn-5/2 Gaussian Process to series cell resistance $R_{\text{cell}}(\sigma)$ with live interactive ionic/electronic/kinetic regime sliders (`w_ion`, `w_elec`, `w_kin`, $\sigma_0/\kappa_0$), proposing the next scan via Expected Improvement.
   - **1-Click NIA Report Bundle (`.zip`):** Exports `manifest.json` (provenance, versions, git SHA, data origin), `metrics.json`, `bpx_parameters.json`, `pybamm_parameters.json`, standalone `reproduce.py`, publication PNGs, and calibrated `microstructure.ome.zarr`.

---

## Run Locally in Under 10 Minutes (`M3.2`)

### Option A — Docker Compose (Zero Local Toolchain Setup)

```bash
git clone https://github.com/BASE-Laboratory/OpenImpala-Hub.git
cd OpenImpala-Hub
docker compose up --build
```

Then open `http://localhost:7860` (healthcheck at `http://localhost:7860/healthz`).

### Option B — Python Virtual Environment (`Linux x86_64` C++/MPI or Universal Fallback)

```bash
git clone https://github.com/BASE-Laboratory/OpenImpala-Hub.git
cd OpenImpala-Hub
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[mpi,dev]"      # Linux x86_64 C++/MPI wheel; on macOS/ARM use: pip install -e ".[dev]"

# Verify Layer 4 boundaries & analytical closed-form sanity gate (tau = 1.0000)
python -m basecoat.layers .
openimpala-hub verify

# Launch interactive web dashboard on port 7860
openimpala-hub serve --host 0.0.0.0 --port 7860
```

---

## CLI & JSON REST API Contract

### Command-Line Interface

```bash
openimpala-hub --help
openimpala-hub presets
openimpala-hub verify
openimpala-hub run --preset carbon_felt_compression --clamping-kpa 180 --rev-study --output-bundle report_180kpa.zip
openimpala-hub run --preset composite_cathode_3phase --electrode-role positive --json
```

### HTTP JSON Endpoints

- `GET /healthz` — Container health, `openimpala` / `basecoat` versions, and active solver backend (`cpp-cpu` or `scipy-pure`).
- `GET /api/v1/presets` — Machine-readable preset catalogue.
- `POST /api/v1/upload` — Stage a `.tif`, `.npy`, `.h5`, or `.ome.zarr.zip` into UUID-isolated storage with zip-slip/size validation and return pre-solve histogram & calibration metadata.
- `POST /api/v1/solve` — Execute a preset (`{"preset": "carbon_felt_compression"}`) or custom `{"microstructure": {...}, "solve": {...}}` payload (enforcing `HUB_MAX_SOLVE_SIDE` and staged-upload verification) and return `SolveRunSummary` JSON.
- `GET /api/v1/bundle/<run_id>.zip` — Stream the reproducible `.zip` NIA report bundle for a cached `run_id`.

---

## Cloud & Serverless Deployment (`M3.1`)

- **Any Container PaaS / VPS (`Dockerfile`):** Listens on `PORT=7860` as non-root `uid=1000` with `/healthz` healthchecks (compatible with Hetzner Cloud + Caddy, Hugging Face PRO Docker Spaces, Render, Railway, AWS App Runner, and Azure Container Apps).
- **Zero-Idle-Cost Serverless + Per-Second GPU Burst (`deploy/modal_app.py`):**
  ```bash
  pip install -e ".[modal]"
  python deploy/modal_app.py --check
  modal deploy deploy/modal_app.py
  ```
  Deploys the Dash WSGI app on Modal's Starter tier (`$0/mo` subscription with `$30/mo` free compute credits, scaling to zero when idle) alongside an on-demand NVIDIA L4 GPU burst function (`gpu_burst_solve`) for `256³` CUDA solves and live `PorousDiff` 3D diffusion checkpoints. See [`docs/hosting_and_qr_case.md`](docs/hosting_and_qr_case.md) for cost benchmarks.

---

## Reproducing the EPSRC NIA Figures & Walkthrough

```bash
python scripts/make_nia_figures.py
python scripts/record_demo_walkthrough.py
```

All generated plots and exact hardware/solver provenance sidecars are written to [`docs/figures/`](docs/figures/) and summarised in [`docs/nia_evidence_pack.md`](docs/nia_evidence_pack.md) and [`docs/seed_impact_summary.md`](docs/seed_impact_summary.md).

---

## Ecosystem Governance & Licence

`OpenImpala-Hub` is a **Layer 4 Service** in the BASE Laboratory 4-layer architecture ([`ECOSYSTEM.md`](https://github.com/BASE-Laboratory/basecoat/blob/main/ECOSYSTEM.md)) and is released under the **BSD 3-Clause License** ([`LICENSE`](LICENSE)).
