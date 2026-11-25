# Cloud Hosting, Serverless GPU & Institutional QR Business Case

*Verified November 2026 against empirical `OpenImpala` container benchmarks and current provider rate cards.*

## 1. Key Technical Finding: Decouple Always-On UI/CPU Solves from Burst GPU Inference

Empirical benchmarking of `openimpala[openmpi]` (compiled C++17 AMReX matrix-free `MLMG` wheel) on a standard 2-vCPU Intel Xeon container (`2.30 GHz`, `7.8 GiB` RAM) shows:

- **Interactive web domains (`48³`–`64³`, 3 axes + 3D potential & flux fields):** **`0.5–1.4 s`** on 2 CPU cores (`< 150 MB` RAM).
- **Medium research domains (`96³`, 3 axes + 3D fields):** **`~5.3 s`** on 2 CPU cores (`~280 MB` RAM).
- **Public web cap (`128³`, 2.1M voxels, 3 axes + 3D fields):** **`~14.9 s`** on 2 CPU cores (`638 MB` solver RAM).

**Conclusion on GPUs:**
- Interactive `OpenImpala` solves up to `96³`–`128³` are **already sub-second to a few seconds on commodity CPUs**—faster than a cold GPU container boots.
- GPU acceleration (`NVIDIA L4 24 GB` / `A10G`) is transformative specifically for **(a) live `PorousDiff` conditional 3D UNet DDIM sampling** (60–100 `96³` 3D UNet passes with classifier-free guidance: ~5 min on 2 CPUs vs ~5 s on L4 GPU) and **(b) `256³` full-REV `openimpala-cuda` solves** (`16.8M` voxels: ~90 s CPU vs ~3–5 s GPU).
- Renting an always-on cloud GPU VM (`AWS g4dn.xlarge` / `g6.xlarge` or `Azure NC4as_T4_v3` at **£305–£580/month**) wastes >99% of spend during idle hours. Instead, **serverless per-second GPU bursting** (`deploy/modal_app.py`) costs **~$0.0022 (<0.2p) per 10-second L4 GPU job** and **$0.00 when idle**.

---

## 2. Hosting Tiers & Annual Budget Comparison

| Tier | Architecture | Monthly Cost | Annual Cost | Reviewer / Classroom Experience |
| :--- | :--- | ---: | ---: | :--- |
| **Tier 0: £0 Prototype (Immediate)** | `deploy/modal_app.py` on **Modal Starter** (`$0/mo` fee + `$30/mo` free CPU & L4 GPU compute credits) + local `docker compose up` | **£0** | **£0** | Zero cost; supports ~10,000 ten-second L4 GPU bursts/month free. Scales to zero after 3 min idle (**~4–8 s wake-up** on first page visit). |
| **Tier 1: Always-Awake Custom Domain** | **Hetzner Cloud VPS** (`CX32` / `CAX31`, 4–8 vCPU, 8–16 GB RAM, Caddy HTTPS on `hub.openimpala.org`) *or* **Hugging Face PRO Docker Space** (`$9/mo`), bursting to Modal's free `$30/mo` L4 GPU pool | **£7 – £18** | **£85 – £220** | **0 s cold-start 24/7** throughout the EPSRC NIA review window; branded domain; zero runaway GPU risk. |
| **Tier 2: Institutional QR Production & Teaching Tier** | Dedicated 8-vCPU / 32 GB cloud VM + custom domain + automated backups + object storage + prepaid serverless L4 GPU burst credits for 30+ concurrent MSc students | **£45 – £80** | **£550 – £950** | Handles simultaneous classroom laboratory cohorts and external national-facility collaborators with guaranteed SLA. |
| *(Avoid)* 24/7 Dedicated Cloud GPU VM | Always-on `AWS g6.xlarge` (L4) / `Azure NC4as_T4_v3` | £305 – £580 | £3,600 – £7,000 | Identical user capability to Tier 1/2 at **10–20× higher cost**. |

> **Note on Hugging Face Spaces (July 2026 policy update):** Free personal Hugging Face accounts no longer include `CPU Basic` Docker Spaces; hosting a Docker/Dash Space on Hugging Face now requires a `$9/mo` (`~£7/mo`) PRO subscription.

---

## 3. Recommended QR Ask (~£750 – £1,000 for 12–18 Months)

With the functional **Tier 0** prototype already built, containerised, and verifiable online, a modest **£750–£1,000 internal QR allocation** upgrades the OpenImpala Hub to **Tier 2** across the entire **EPSRC New Investigator Award** and **ERC Starting Grant** review cycle while supporting hands-on MSc image-based battery modelling practicals at the University of Greenwich without local student software installation.
