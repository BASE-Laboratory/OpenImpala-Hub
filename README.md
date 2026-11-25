---
title: The OpenImpala Hub
emoji: 🔋
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
license: bsd-3-clause
---

# The OpenImpala Hub (`v0.3.0`)

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Layer 4: Service](https://img.shields.io/badge/BASE_Lab-Layer_4_Service-0d9488)](AGENTS.md)
[![OpenImpala](https://img.shields.io/badge/Solver-OpenImpala_v5.x-2563eb)](https://github.com/BASE-Laboratory/OpenImpala)
[![SMS Mechanics](https://img.shields.io/badge/Mechanics-SMS_Fortran_3D-7c3aed)](https://github.com/BASE-Laboratory/structural_mechanics_solver)
[![Lite Digital Twin](https://img.shields.io/badge/Live_Demo-GitHub_Pages-16a34a)](https://base-laboratory.github.io/OpenImpala-Hub/)

**The OpenImpala Hub** is an integrated, browser-accessible electro-chemo-mechanical platform coupling **3D microstructure synthesis & virtual compression**, **AMReX/HYPRE transport physics (`OpenImpala`)**, **3D finite-volume elasticity (`SMS`)**, **C-rate cell electrochemistry**, **out-of-core volumetric visualisation**, and **autonomous experiment steering** in a zero-installation workflow.

- **Always-On Lite Digital Twin (£0/mo GitHub Pages):** [https://base-laboratory.github.io/OpenImpala-Hub/](https://base-laboratory.github.io/OpenImpala-Hub/)

Developed by the **[BASE Laboratory](https://github.com/BASE-Laboratory)** at the University of Greenwich (CSEG ECR Seed Award prototype supporting the EPSRC New Investigator Award & *Heuristic Operando* digital-twin programme).

- **Maintainer:** Dr. James Le Houx (`j.lehoux@greenwich.ac.uk`)
- **Contributors:** James Le Houx (PI, solver & digital-twin engine), Ali Humza Butt (volumetric web dashboard research strand), Asif Ali (3D diffusion microstructure synthesis research strand)

---

## Four Multiphysics Pillars + Autonomous Steering (`v0.3.0`)

1. **3D Synthesis, Ingestion & Morphological Validation (`generators`, `porousdiff_bridge`, `morphology`)**
   - **Procedural & Diffusion Sources:** Non-woven fibrous **Carbon Felt** with mechanical Z-compression (`kPa` $\rightarrow$ $\epsilon_z$) and surface-supported catalyst/binder shells, `openimpala.synthetic` (`gaussian_blobs`, `overlapping_spheres`, `multiphase_electrode`), analytical verification domains, and drop-in [`PorousDiff`](https://github.com/BASE-Laboratory/PorousDiff) `.ome.zarr` / conditional UNet checkpoints.
   - **Out-of-Core Experimental XCT / FIB-SEM + Virtual Calendering:** Region-lazy `.ome.zarr.zip`, `.tif`, `.npy`, and `.h5` ingestion via [`basecoat`](https://github.com/BASE-Laboratory/basecoat) (`v0.3`) with live Otsu/manual segmentation preview, calibrated anisotropic `(dz, dy, dx)` SI micrometre spacing, and **solid-mass-conserving kinematic virtual Z-compression** (`--virtual-compress-strain`).
   - **Quantitative Stereology & Pore Network Diagnostics:** Exact 6-connected **Percolating vs Dead-End vs Isolated/Trapped** pore partition, 3D FFT Wiener–Khinchin two-point correlation $S_2(r)$ ($S_2(0) = \varepsilon$), inscribed pore/throat diameter PSD (`D10/D50/D90`), and directional chord-length anisotropy.
2. **Coupled Layer-3 Transport & 3D Structural Mechanics (`solver`, `mechanics`)**
   - **Transport (`OpenImpala` C++/AMReX/HYPRE):** 3-axis tortuosity $(\tau_x, \tau_y, \tau_z)$, effective diffusivity, composite multiphase solid wiring, $3\times 3$ periodic homogenisation tensors, subvolume **REV** convergence (`openimpala.rev_study`), unbiased $2/3$ Cauchy–Crofton SSA, 3D potential $\phi(z,y,x)$, and diffusive flux crowding $|\mathbf{J}/J_0|$.
   - **3D Elasticity (`SMS` Fortran MAC-FV):** Optional `--mechanics` solve computing homogenised effective Young's modulus $E_{\text{eff}}/E_0$ and cell-centred von Mises stress $\sigma_{\text{vM}}(z,y,x)$ for direct slice co-visualisation alongside $|\mathbf{J}/J_0|$.
3. **Microstructure-to-Cell C-Rate Electrochemistry (`electrochem`)**
   - Predicts electrolyte salt-depletion limiting current $J_{\text{lim}}$ and critical C-rate $C_{\text{lim}}$, `0.5C–4C` galvanostatic discharge curves $V(Q)$, and stacked ionic / electronic / Butler–Volmer kinetic / concentration overpotential waterfalls across `NMC811`, `LFP`, and `Vanadium Redox Felt` chemistries, alongside role-aware **PyBaMM / Faraday BPX** continuum exports.
4. **Interactive Browser Twin, Persistent Campaigns & Greenwich Teaching Lab (`app`)**
   - **6-Tab Workspace + `🎓 Teaching Lab` Drawer:** 2D Flux/Potential/Phase/von-Mises slices, 3D marching-cubes cutaway + Zoom-ROI wireframes, 2×2 Morphology validator, Cell Electrochemistry & Mechanics, Side-by-Side State Comparator, and live Matérn-5/2 Gaussian Process planner with **1-click 5-Point Campaign Sweeps**, `localStorage` persistence, `.json` campaign save/load, and a 4-exercise MSc/BEng Teaching Lab with auto-graded `PASS` badges.
   - **1-Click NIA Report Bundle (`.zip`):** Exports `manifest.json`, `metrics.json`, `bpx_parameters.json`, `pybamm_parameters.json`, standalone `reproduce.py`, publication PNGs (`transport_slice_and_metrics.png`, `autonomous_gp_planner.png`, `morphology_and_electrochem.png`), and calibrated `microstructure.ome.zarr`.

---

## Run Locally in Under 10 Minutes (`M3.2`)

### Option A — Docker Compose (Includes C++ `OpenImpala` + Compiled Fortran `SMS`)

```bash
git clone https://github.com/BASE-Laboratory/OpenImpala-Hub.git
cd OpenImpala-Hub
docker compose up --build
```

Then open `http://localhost:7860` (healthcheck at `http://localhost:7860/healthz`).

### Option B — Python Virtual Environment (`Linux x86_64` C++/MPI + Optional Fortran `SMS`)

```bash
git clone https://github.com/BASE-Laboratory/OpenImpala-Hub.git
cd OpenImpala-Hub
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[mpi,sms,dev]"  # omit ,sms if gfortran is not installed; on macOS/ARM use: pip install -e ".[dev]"

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
openimpala-hub run --preset carbon_felt_compression --clamping-kpa 180 --mechanics --chemistry vanadium_flow_felt --output-bundle report_180kpa.zip
openimpala-hub run --preset composite_cathode_3phase --electrode-role positive --virtual-compress-strain 0.20 --chemistry nmc811_li_ion --json
```

### HTTP JSON Endpoints

- `GET /healthz` — Container health, `openimpala` / `basecoat` versions, and active solver backend (`cpp-cpu` or `scipy-pure`).
- `GET /api/v1/presets` — Machine-readable preset catalogue.
- `POST /api/v1/upload` — Stage a `.tif`, `.npy`, `.h5`, or `.ome.zarr.zip` into UUID-isolated storage with zip-slip/size validation and return pre-solve histogram & calibration metadata.
- `POST /api/v1/solve` — Execute a preset (`{"preset": "carbon_felt_compression"}`) or custom `{"microstructure": {...}, "solve": {...}}` payload (enforcing `HUB_MAX_SOLVE_SIDE` and staged-upload verification) and return `SolveRunSummary` JSON (including `morphology`, `electrochem`, and `mechanics`).
- `GET /api/v1/bundle/<run_id>.zip` — Stream the reproducible `.zip` NIA report bundle for a cached `run_id`.

---

## Cloud, GitHub Pages & Serverless GPU Deployment (`M3.1`)

- **Always-On Lite Digital Twin (£0/month GitHub Pages):** Automatically deployed from [`docs_site/`](docs_site/) via [`.github/workflows/pages.yml`](.github/workflows/pages.yml) to `https://base-laboratory.github.io/OpenImpala-Hub/`.
- **Any Container PaaS / VPS (`Dockerfile`):** Listens on `PORT=7860` as non-root `uid=1000` with `/healthz` healthchecks (Hetzner Cloud + Caddy, Hugging Face PRO Docker Spaces, Render, Railway, AWS App Runner, Azure Container Apps).
- **Zero-Idle-Cost Serverless + Per-Second L4 GPU Burst (`deploy/modal_app.py`):**
  ```bash
  pip install -e ".[modal]"
  python deploy/modal_app.py --check
  modal deploy deploy/modal_app.py
  ```
  Deploys the Dash WSGI app on Modal's Starter tier (`$0/mo` subscription with `$30/mo` free compute credits, scaling to zero when idle) alongside an on-demand NVIDIA L4 GPU burst worker (`gpu_burst_solve`) with `SMS` + `PorousDiff` (`feature/paper-ready-conditional`) pre-installed. See [`docs/hosting_and_qr_case.md`](docs/hosting_and_qr_case.md) for cost benchmarks.

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
