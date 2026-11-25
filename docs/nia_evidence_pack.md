# EPSRC New Investigator Award (NIA) — Preliminary Feasibility Evidence Pack

*Generated from verified executions of `BASE-Laboratory/OpenImpala-Hub` (`v0.1.0`, `openimpala==5.0.0` `cpp-cpu` AMReX MLMG, `basecoat==0.3.0`) on a 2-vCPU Intel Xeon (`2.30 GHz`, `7.8 GiB` RAM) container. Full numerical arrays and provenance metadata are archived in [`figures/benchmark_provenance.json`](figures/benchmark_provenance.json).*

---

## 1. De-Risked Technical Feasibility for the EPSRC NIA

The CSEG ECR Seed prototype was designed to retire the three primary technical risks prior to submitting the planned EPSRC New Investigator Award (*Autonomous Experimental Steering for Operando X-Ray Tomography*):

1. **Cross-Language HPC-to-Python Coupling (Retired — WP1):**
   `OpenImpala` has been rewritten from legacy C++/Fortran CLI executables into portable C++17 AMReX `ParallelFor` kernels exposed via `pybind11` (`openimpala`, `openimpala-cuda` wheels on PyPI). Matrix-free geometric multigrid (`AMReX MLMG`) solves full 3-axis steady-state elliptic PDEs *plus* 3D potential and harmonic-mean diffusive flux fields in **`1.37 s` at `64³`**, **`4.75 s` at `96³`**, and **`13.44 s` at `128³` (`2.1M` voxels)** on a modest 2-vCPU container (`808.5 MiB` peak RSS)—outperforming pure-Python `SciPy CG` by **`6.3×` (`64³`) to `8.0×` (`80³`)** and enabling real-time in-the-loop steering during synchrotron beamtime.
2. **Standardised Calibrated Volumetric Data Contract (Retired — Layer 1–4 Architecture):**
   All tools and solvers exchange calibrated cell-centred `(z, y, x)` `Volume` / `VolumeSet` containers and OME-Zarr v0.4 stores (`basecoat v0.3`), enforcing explicit SI micrometre-to-metre unit conversion at solver boundaries and exporting continuum parameters directly in **PyBaMM** and **Faraday Institution BPX** formats.
3. **Closed-Loop Digital Twin Steering Prototype (Retired — WP3):**
   `OpenImpala-Hub` demonstrates both **spatial beamline steering** (ranking non-overlapping 3D sub-volumes `ZoomROI` with peak current crowding $|\mathbf{J}/J_0|$ and throat constriction for targeted zoom XCT / XRD-CT) and **parameter-space Bayesian steering** (Matérn-5/2 Gaussian Process Expected-Improvement acquisition over series cell resistance $R_{\text{cell}}(\sigma)$).

---

## 2. Preliminary Benchmark Figures (Synthetic Microstructure Provenance)

> **Data Provenance Note:** Per BASE Lab reporting rules, all three preliminary figures below are computed on deterministic **synthetic** 3D microstructures (`int32` phase grids, `float64` PDE solves, `float32` flux fields) via `python scripts/make_nia_figures.py`. Drop-in `.ome.zarr` integration for Isambard-AI `PorousDiff` 3D diffusion realisations and synchrotron XCT tomograms is live via `data/porousdiff/` and `openimpala-hub run --input`.

### Figure 1 — Solver Latency & `pybind11` C++ Speedup (`figures/fig1_solver_scaling.png`)
![Solver Latency Scaling and Speedup](figures/fig1_solver_scaling.png)

- **Key numbers for grant text:** On 2 Xeon vCPUs, 3-axis C++ `AMReX MLMG` (+ 3D flux fields) completes in `0.83 s` (`48³`), `1.37 s` (`64³`), `2.77 s` (`80³`), `4.75 s` (`96³`), and `13.44 s` (`128³`), achieving a **`6.3×`–`8.0×` speedup** over `SciPy CG` (`22.30 s` at `80³`) within `< 1 GiB` container memory.

### Figure 2 — Anisotropic Fibrous Electrode Compression Trade-Off (`figures/fig2_anisotropic_compression_tradeoff.png`)
![Anisotropic Carbon Felt Compression Trade-Off](figures/fig2_anisotropic_compression_tradeoff.png)

- **Key numbers for grant text:** Across `56³` non-woven fibrous carbon-felt mats (`n = 3` stochastic seeds, `1.6 µm/voxel`), mechanical Z-clamping (`25` $\rightarrow$ `480 kPa`) reproduces through-plane vs in-plane tortuosity anisotropy ($\tau_Z = 1.33 \rightarrow 3.16$ vs $\tau_{XY} = 1.20 \rightarrow 2.35$) and identifies an interior electrochemical clamping sweet spot around **`σ* ≈ 130 kPa`** ($R_{\text{cell}} = 2.30 \pm 0.10$, $\varepsilon = 0.73$), balancing low-load Z-fibre contact resistance (`2.80` at `25 kPa`) against high-load pore pinch-off (`8.27` at `480 kPa`, where $D_{\text{eff},Z}/D_0$ drops from `0.61` to `0.15`).

### Figure 3 — Active Bayesian Experiment Steering vs Passive Scan Schedules (`figures/fig3_autonomous_steering_efficiency.png`)
![Autonomous Bayesian Steering Efficiency](figures/fig3_autonomous_steering_efficiency.png)

- **Key numbers for grant text:** Benchmarked against a 17-scan (`20..500 kPa`) `OpenImpala` ground-truth campaign (optimum `110 kPa`, $R_{\text{cell}} = 2.342$), active Matérn-5/2 Gaussian Process steering reduces optimality regret by **`3.1×` at 3 scans** (`0.266` vs `0.821` passive uniform grid), reaches near-zero regret (`0.0047`) within **5 scans** (`[20, 500, 80, 170, 140] kPa`), and locates the exact discrete optimum by **6 scans**—quantifying *"scientific insight per photon"* for synchrotron beamtime allocation.
