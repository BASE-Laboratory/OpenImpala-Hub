# CSEG ECR Seed Funding — Deliverables & Impact Summary

- **Project Title:** *The OpenImpala Hub*
- **PI:** Dr. James Le Houx (`CSEG`, University of Greenwich)
- **Postgraduate Student Strands:** Ali Humza Butt (volumetric web dashboard), Asif Ali (3D generative diffusion surrogates on Isambard-AI GH200)
- **Software Repositories:**
  - Layer 4 Integrated Hub: <https://github.com/BASE-Laboratory/OpenImpala-Hub> (`BSD-3-Clause`)
  - Layer 3 Transport Solver: <https://github.com/BASE-Laboratory/OpenImpala> (`v5.0` / `v5.1`, PyPI `openimpala` & `openimpala-cuda`)
  - Layer 2 3D Diffusion Surrogate: <https://github.com/BASE-Laboratory/PorousDiff> (`main` MSc archive + `feature/paper-ready-conditional`)
  - Layer 1 Core Volumetric Contract: <https://github.com/BASE-Laboratory/basecoat> (`v0.3.0`)

---

## 1. Work Package Completion Summary

| Work Package | Planned Deliverable | Delivered Outcome |
| :--- | :--- | :--- |
| **WP1: Core Solver Refactoring & API Engineering** | Modular multiphase C++ backend + `pybind11` Python bindings | **Completed & exceeded:** Legacy Fortran 90 replaced by single-source C++17 AMReX CPU/CUDA kernels; matrix-free `AMReX MLMG` (`3.2×–4.4×` lower memory); composite multiphase transport & $3\times 3$ periodic homogenisation; pre-built Linux CPU/CUDA and universal PyPI wheels (`pip install openimpala`); JOSS manuscript prepared. |
| **WP2: Generative Modelling & Volumetric Visualisation** | 3D generative model on electrode tomograms (AA) + browser 3D/slice viewer with out-of-core chunking (AB) | **Completed:** (1) `PorousDiff` trained 3D autoencoder/LPIPS and conditional 3D DDPM/DDIM architectures on Isambard-AI GH200 (`10,000 GPUhr` allocation), plus instant zero-GPU procedural fibrous carbon-felt synthesis; (2) `BatteryExplorer` (AB dissertation) & `OpenImpala-Hub` deliver out-of-core slice streaming, marching-cubes 3D cutaways, and diffusive flux crowding overlays in the browser. |
| **WP3: Cloud Containerisation, Digital Twin & Export** | Unified zero-install web platform + autonomous experiment-planning heuristic + reproducible report bundle | **Completed:** `OpenImpala-Hub` unifies generation/upload, `OpenImpala` PDE solves, 2D/3D visualisation, 3D synchrotron Zoom-ROI bottleneck detection, 1D Bayesian Gaussian Process experiment steering, and 1-click `.zip` NIA/BPX report bundles inside a portable Docker + Modal serverless stack. |

---

## 2. Key Poster & Impact Highlights (ECR Network Event)

- **Zero-Installation Research & Teaching Platform:** Students and external collaborators can launch the entire pipeline in a browser (or locally in `<10 min` via `docker compose up`) with built-in analytical verification (`open_cube` $\tau = 1.0000$, $|\mathbf{J}/J_0| = 1.000$) and 3-phase composite battery presets.
- **Interactive HPC Performance on Commodity Containers:** Matrix-free C++ `AMReX MLMG` via `pybind11` solves 3-axis transport + 3D flux fields in **`1.37 s` (`64³`)** and **`4.75 s` (`96³`)** on 2 CPU cores (`6.3×–8.0×` faster than pure Python), eliminating mandatory 24/7 GPU hosting costs.
- **Onward Grant Leverage:** Directly underpins the **EPSRC New Investigator Award** (*Autonomous Operando Steering* with STFC / Diamond Light Source collaborators) and establishes the BASE Laboratory's modular 4-layer open-source software ecosystem.
