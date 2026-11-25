"""Autonomous experiment steering heuristics (WP3 Digital Twin).

Implements two complementary decision engines aligned with the BASE Lab
Heuristic Operando framework:
1. **3D Spatial Zoom-ROI Trigger (`find_flux_bottleneck_rois`)**:
   Computes the cell-centred dimensionless diffusive flux field
   `|J_tilde(z, y, x)| = |-D grad(phi)| / (1 / L_dir)` using exact harmonic-mean
   inter-cell conductances (matching `OpenImpala`'s finite-volume discretisation,
   so `|J_tilde| == 1.0` everywhere in a uniform open domain). Uses 3D integral
   sums (`O(N_vox)`) to rank non-overlapping sub-volumes where current crowding
   and pore constriction peak, outputting calibrated micrometre coordinates for
   targeted high-resolution synchrotron zoom XCT / XRD-CT.
2. **1D Bayesian Gaussian Process Setpoint Planner (`suggest_next_setpoint_gp`)**:
   Models total static-cell series resistance
       `R_cell(sigma) = w_ion / D_tilde_Z + w_elec / sigma_tilde_Z + w_kin / A_tilde`
   across compression / porosity runs using a pure-SciPy Cholesky Matérn-5/2
   Gaussian Process with marginal-likelihood length-scale selection, proposing
   the next experimental setpoint via Expected Improvement (EI).
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import erf
from basecoat import Volume

from ..contracts import BayesianSetpointSuggestion, DirectionName, ZoomROI


def compute_cell_resistance(
    *,
    d_eff_z: float,
    sigma_eff_z: float,
    active_area_m_inv: float,
    thickness_ratio: float = 1.0,
    ref_area_m_inv: float = 1.2e5,
    sigma_over_kappa_bulk: float = 16.0,
    w_ion: float = 0.58,
    w_elec: float = 0.27,
    w_kin: float = 0.15,
    regularisation_floor: float = 0.004,
) -> float:
    """Compute dimensionless series cell resistance `R_cell` balancing ionic, electronic, and kinetic losses.

    Follows the series-resistance formulation in `PorousDiff` (`porousdiff_workflows/optimization.py`):
    - Over-compression pinches pore throats (`d_eff_z -> 0`), spiking ionic mass-transport & salt-depletion loss.
    - Under-compression leaves sparse Z-fibre contacts (`sigma_eff_z -> 0`), spiking electronic contact loss.
    - Moderate compression (`~150-260 kPa`) wires the 3D carbon network (`sigma_bulk >> kappa_bulk`) while
      preserving open through-plane ionic pathways and high Cauchy-Crofton active area `A_tilde`.
    """
    d_tilde = max(float(d_eff_z), regularisation_floor)
    sig_tilde = max(float(sigma_eff_z) * float(sigma_over_kappa_bulk), regularisation_floor * 4.0)
    a_tilde = max(float(active_area_m_inv) / max(float(ref_area_m_inv), 1.0), regularisation_floor)
    l_ratio = max(float(thickness_ratio), 0.22)

    # Mass-transport / Nernst depletion scales inversely with D_eff,Z^1.35 as throats pinch shut
    r_ion = float(w_ion) / (d_tilde**1.35)
    r_elec = float(w_elec) * l_ratio / sig_tilde
    r_kin = float(w_kin) / a_tilde
    return float(r_ion + r_elec + r_kin)


def compute_diffusive_flux_field(
    phase_vol: Volume,
    potential_field: np.ndarray,
    *,
    flow_direction: DirectionName = "z",
    conducting_phase: int = 0,
    phase_diffusivities: Mapping[int, float] | None = None,
) -> np.ndarray:
    """Compute cell-centred dimensionless flux magnitude `|J_tilde(z, y, x)|` (`float32`).

    Uses harmonic-mean face diffusivities `D_face = 2 D_L D_R / (D_L + D_R)` and exterior
    half-cell Dirichlet boundaries (`phi = 0` at inlet face, `phi = 1` at outlet face along
    `flow_direction`) normalised by the macroscopic imposed gradient `1 / L_flow`.
    In a uniform single-phase cube (`D = 1`), returns `1.0` at every voxel to machine precision.
    """
    arr = np.asarray(phase_vol.data)
    phi = np.asarray(potential_field, dtype=np.float64)
    if arr.shape != phi.shape or arr.ndim != 3:
        raise ValueError(
            f"Phase volume shape {arr.shape} and potential shape {phi.shape} must match and be 3D."
        )

    dz, dy, dx = phase_vol.spacing_in("m")
    spacings = (dz, dy, dx)
    dir_to_axis = {"z": 0, "y": 1, "x": 2}
    flow_ax = dir_to_axis[flow_direction]
    l_flow = float(arr.shape[flow_ax]) * spacings[flow_ax]
    grad_macro = 1.0 / max(l_flow, 1e-30)

    if phase_diffusivities is not None:
        d_cell = np.zeros(arr.shape, dtype=np.float64)
        for pid, dval in phase_diffusivities.items():
            d_cell[arr == int(pid)] = max(0.0, float(dval))
    else:
        d_cell = np.where(arr == int(conducting_phase), 1.0, 0.0).astype(np.float64)

    # Zero out isolated/dead-end conducting clusters not 6-connected to both inlet and outlet
    # (matches OpenImpala TortuositySolverBase flood-fill activity mask, preventing wall jumps on dead pores)
    from scipy import ndimage as _ndimage

    cond_mask = d_cell > 0.0
    if np.any(cond_mask) and not np.all(cond_mask):
        lbl, _ = _ndimage.label(cond_mask)
        sl_in = [slice(None)] * 3
        sl_out = [slice(None)] * 3
        sl_in[flow_ax] = 0
        sl_out[flow_ax] = arr.shape[flow_ax] - 1
        inlet_ids = set(int(v) for v in np.unique(lbl[tuple(sl_in)]) if v > 0)
        outlet_ids = set(int(v) for v in np.unique(lbl[tuple(sl_out)]) if v > 0)
        active_ids = inlet_ids & outlet_ids
        if active_ids:
            active_mask = np.isin(lbl, list(active_ids))
            d_cell = np.where(active_mask, d_cell, 0.0)
        else:
            d_cell.fill(0.0)

    j_sq = np.zeros(arr.shape, dtype=np.float64)

    for ax in range(3):
        h = spacings[ax]
        n_ax = arr.shape[ax]
        # Face array has length n_ax + 1 along `ax`
        face_shape = list(arr.shape)
        face_shape[ax] = n_ax + 1
        j_face = np.zeros(face_shape, dtype=np.float64)

        sl_left = [slice(None)] * 3
        sl_right = [slice(None)] * 3
        sl_int = [slice(None)] * 3
        sl_left[ax] = slice(0, n_ax - 1)
        sl_right[ax] = slice(1, n_ax)
        sl_int[ax] = slice(1, n_ax)

        dl = d_cell[tuple(sl_left)]
        dr = d_cell[tuple(sl_right)]
        denom = dl + dr
        d_harm = np.divide(2.0 * dl * dr, denom, out=np.zeros_like(denom), where=denom > 0.0)
        j_face[tuple(sl_int)] = -d_harm * (phi[tuple(sl_right)] - phi[tuple(sl_left)]) / h

        if ax == flow_ax:
            # Exterior Dirichlet boundaries: inlet phi_wall=0 at -h/2, outlet phi_wall=1 at +h/2
            sl_0 = [slice(None)] * 3
            sl_end = [slice(None)] * 3
            sl_f0 = [slice(None)] * 3
            sl_fend = [slice(None)] * 3
            sl_0[ax] = 0
            sl_end[ax] = n_ax - 1
            sl_f0[ax] = 0
            sl_fend[ax] = n_ax
            # Note: OpenImpala sets vlo=0 or vhi=1 depending on inlet/outlet sign; detect orientation from mean slice
            mean_first = float(np.mean(phi[tuple(sl_0)]))
            mean_last = float(np.mean(phi[tuple(sl_end)]))
            if mean_last >= mean_first:
                wall_lo, wall_hi = 0.0, 1.0
            else:
                wall_lo, wall_hi = 1.0, 0.0
            j_face[tuple(sl_f0)] = -d_cell[tuple(sl_0)] * (phi[tuple(sl_0)] - wall_lo) / (0.5 * h)
            j_face[tuple(sl_fend)] = -d_cell[tuple(sl_end)] * (wall_hi - phi[tuple(sl_end)]) / (0.5 * h)

        # Average adjacent face fluxes onto cell centres
        sl_fa = [slice(None)] * 3
        sl_fb = [slice(None)] * 3
        sl_fa[ax] = slice(0, n_ax)
        sl_fb[ax] = slice(1, n_ax + 1)
        j_cell_ax = 0.5 * (j_face[tuple(sl_fa)] + j_face[tuple(sl_fb)]) / grad_macro
        j_sq += j_cell_ax * j_cell_ax

    j_mag = np.sqrt(j_sq)
    j_mag[d_cell <= 0.0] = 0.0
    return j_mag.astype(np.float32)


def _box_sum_3d(arr: np.ndarray, win: tuple[int, int, int]) -> np.ndarray:
    """Exact 3D sliding-window sum of shape `(Nz-wz+1, Ny-wy+1, Nx-wx+1)` via integral volumes."""
    wz, wy, wx = win
    padded = np.pad(arr.astype(np.float64), ((1, 0), (1, 0), (1, 0)), mode="constant")
    ii = padded.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    return (
        ii[wz:, wy:, wx:]
        - ii[:-wz, wy:, wx:]
        - ii[wz:, :-wy, wx:]
        - ii[wz:, wy:, :-wx]
        + ii[:-wz, :-wy, wx:]
        + ii[:-wz, wy:, :-wx]
        + ii[wz:, :-wy, :-wx]
        - ii[:-wz, :-wy, :-wx]
    )


def find_flux_bottleneck_rois(
    phase_vol: Volume,
    flux_magnitude: np.ndarray,
    *,
    conducting_phase: int = 0,
    roi_side: int | tuple[int, int, int] = 16,
    top_k: int = 3,
) -> list[ZoomROI]:
    """Identify up to `top_k` non-overlapping 3D sub-volumes exhibiting peak transport bottlenecks."""
    nz, ny, nx = phase_vol.shape
    if isinstance(roi_side, int):
        win = (
            min(max(4, roi_side), nz),
            min(max(4, roi_side), ny),
            min(max(4, roi_side), nx),
        )
    else:
        win = (
            min(max(4, int(roi_side[0])), nz),
            min(max(4, int(roi_side[1])), ny),
            min(max(4, int(roi_side[2])), nx),
        )

    win_vol = float(win[0] * win[1] * win[2])
    pore_mask = (phase_vol.data == int(conducting_phase)).astype(np.float64)
    global_porosity = max(float(np.mean(pore_mask)), 1e-6)

    flux_sum = _box_sum_3d(flux_magnitude, win)
    flux_sq_sum = _box_sum_3d(np.square(flux_magnitude.astype(np.float64)), win)
    pore_sum = _box_sum_3d(pore_mask, win)

    local_porosity = pore_sum / win_vol
    # Require at least 5% conducting voxels in the window so pure solid blocks are ignored
    valid = pore_sum >= max(4.0, 0.05 * win_vol)
    mean_pore_flux = np.divide(
        flux_sum, pore_sum, out=np.zeros_like(flux_sum), where=pore_sum > 0.0
    )
    rms_pore_flux = np.sqrt(
        np.divide(flux_sq_sum, pore_sum, out=np.zeros_like(flux_sq_sum), where=pore_sum > 0.0)
    )
    constriction = np.clip(global_porosity / np.maximum(local_porosity, 0.02), 0.5, 6.0)
    score = np.where(valid, rms_pore_flux * np.sqrt(constriction), -1.0)

    ax_z, ax_y, ax_x = phase_vol.axes
    dz_um, dy_um, dx_um = phase_vol.spacing_in("um")
    oz_um, oy_um, ox_um = phase_vol.origin_in("um")

    rois: list[ZoomROI] = []
    work_score = score.copy()
    for rank in range(1, int(top_k) + 1):
        best_flat = int(np.argmax(work_score))
        best_val = float(work_score.flat[best_flat])
        if best_val <= 0.0:
            break
        z0, y0, x0 = (int(c) for c in np.unravel_index(best_flat, work_score.shape))
        z1, y1, x1 = z0 + win[0], y0 + win[1], x0 + win[2]

        sub_flux = flux_magnitude[z0:z1, y0:y1, x0:x1]
        peak_j = float(np.max(sub_flux))
        mean_j = float(mean_pore_flux[z0, y0, x0])
        loc_eps = float(local_porosity[z0, y0, x0])
        c_idx = float(constriction[z0, y0, x0])

        cz_um = oz_um + (0.5 * (z0 + z1 - 1)) * dz_um
        cy_um = oy_um + (0.5 * (y0 + y1 - 1)) * dy_um
        cx_um = ox_um + (0.5 * (x0 + x1 - 1)) * dx_um
        ext_um = (win[0] * dz_um, win[1] * dy_um, win[2] * dx_um)

        rois.append(
            ZoomROI(
                rank=rank,
                bounds_zyx=((z0, z1), (y0, y1), (x0, x1)),
                center_um_zyx=(round(cz_um, 2), round(cy_um, 2), round(cx_um, 2)),
                extent_um_zyx=(round(ext_um[0], 2), round(ext_um[1], 2), round(ext_um[2], 2)),
                mean_flux_crowding=round(mean_j, 4),
                peak_flux_magnitude=round(peak_j, 4),
                local_porosity=round(loc_eps, 4),
                constriction_index=round(c_idx, 3),
                rationale=(
                    f"Rank #{rank} current-crowding throat: peak |J/J0|={peak_j:.2f}x "
                    f"(mean {mean_j:.2f}x) at local porosity {loc_eps:.2f} "
                    f"(constriction {c_idx:.2f}x vs bulk {global_porosity:.2f})."
                ),
            )
        )

        # Suppress overlapping windows
        sz0 = max(0, z0 - win[0] + 1)
        sz1 = min(work_score.shape[0], z1)
        sy0 = max(0, y0 - win[1] + 1)
        sy1 = min(work_score.shape[1], y1)
        sx0 = max(0, x0 - win[2] + 1)
        sx1 = min(work_score.shape[2], x1)
        work_score[sz0:sz1, sy0:sy1, sx0:sx1] = -1.0

    return rois


def _matern52_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float) -> np.ndarray:
    """Matérn-5/2 covariance matrix between 1D arrays `x1` and `x2`."""
    d = np.abs(x1[:, None] - x2[None, :]) / max(float(length_scale), 1e-6)
    sqrt5_d = math.sqrt(5.0) * d
    return (1.0 + sqrt5_d + (5.0 / 3.0) * (d * d)) * np.exp(-sqrt5_d)


def suggest_next_setpoint_gp(
    observed_x: Sequence[float],
    observed_r_cell: Sequence[float],
    *,
    bounds: tuple[float, float] = (20.0, 500.0),
    parameter_name: str = "Clamping pressure",
    parameter_unit: str = "kPa",
    n_grid: int = 121,
    xi: float = 0.01,
) -> BayesianSetpointSuggestion:
    """Fit a 1D Matérn-5/2 Gaussian Process to `(x, R_cell)` and propose `x_next` via Expected Improvement."""
    x_obs = np.asarray(observed_x, dtype=np.float64)
    y_obs = np.asarray(observed_r_cell, dtype=np.float64)
    if x_obs.size == 0 or x_obs.size != y_obs.size:
        raise ValueError("observed_x and observed_r_cell must be non-empty and have equal length.")

    lo, hi = float(bounds[0]), float(bounds[1])
    span = max(hi - lo, 1e-6)
    grid = np.linspace(lo, hi, int(n_grid), dtype=np.float64)

    # Normalise input coordinate to [0, 1] and centre output for conditioning stability
    xn = (x_obs - lo) / span
    gn = (grid - lo) / span
    y_mean = float(np.mean(y_obs))
    y_std = float(np.std(y_obs))
    scale_y = y_std if y_std > 1e-6 else max(0.15 * abs(y_mean), 0.05)
    yn = (y_obs - y_mean) / scale_y

    # Grid-search Matérn-5/2 length-scale by exact log marginal likelihood
    best_lml = -float("inf")
    best_ls = 0.28
    best_noise = 1e-3
    n_pts = xn.size
    eye = np.eye(n_pts, dtype=np.float64)

    if n_pts >= 2:
        for ls in (0.14, 0.22, 0.32, 0.45, 0.65):
            for noise_var in (5e-4, 2e-3, 1e-2):
                k_mat = _matern52_kernel(xn, xn, ls) + noise_var * eye
                try:
                    c_fac, lower = cho_factor(k_mat, lower=True)
                except np.linalg.LinAlgError:
                    continue
                alpha = cho_solve((c_fac, lower), yn)
                log_det = 2.0 * float(np.sum(np.log(np.diag(c_fac))))
                lml = -0.5 * float(yn @ alpha) - 0.5 * log_det - 0.5 * n_pts * math.log(2.0 * math.pi)
                if lml > best_lml:
                    best_lml = lml
                    best_ls = ls
                    best_noise = noise_var

    k_obs = _matern52_kernel(xn, xn, best_ls) + best_noise * eye
    c_fac, lower = cho_factor(k_obs, lower=True)
    alpha = cho_solve((c_fac, lower), yn)

    k_star = _matern52_kernel(gn, xn, best_ls)
    mu_norm = k_star @ alpha
    v = cho_solve((c_fac, lower), k_star.T)
    var_norm = np.maximum(1.0 - np.sum(k_star * v.T, axis=1), 1e-9)
    std_norm = np.sqrt(var_norm)

    mu = y_mean + scale_y * mu_norm
    sigma = scale_y * std_norm

    best_idx = int(np.argmin(y_obs))
    y_best = float(y_obs[best_idx])
    x_best = float(x_obs[best_idx])

    # Expected Improvement for minimisation: EI(x) = E[max(0, y_best - xi - f(x))]
    imp = (y_best - float(xi) * scale_y) - mu
    z = np.divide(imp, sigma, out=np.zeros_like(imp), where=sigma > 1e-9)
    cdf = 0.5 * (1.0 + erf(z / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    ei = np.where(sigma > 1e-9, imp * cdf + sigma * pdf, 0.0)
    ei = np.maximum(ei, 0.0)

    # Penalise points already within 1.5% of an observed setpoint so planner explores genuinely new scans
    min_dist = np.min(np.abs(gn[:, None] - xn[None, :]), axis=1)
    acquisition = np.where(min_dist < 0.015, -1.0, ei)
    if np.all(acquisition <= 0.0):
        # Pure exploration fallback when EI vanishes everywhere
        acquisition = np.where(min_dist < 0.015, -1.0, sigma)

    next_idx = int(np.argmax(acquisition))
    x_next = float(grid[next_idx])
    pred_mean = float(mu[next_idx])
    pred_std = float(sigma[next_idx])
    ei_val = float(ei[next_idx])

    return BayesianSetpointSuggestion(
        parameter_name=parameter_name,
        parameter_unit=parameter_unit,
        suggested_value=round(x_next, 2),
        predicted_r_cell_mean=round(pred_mean, 4),
        predicted_r_cell_std=round(pred_std, 4),
        expected_improvement=round(ei_val, 5),
        current_best_parameter=round(x_best, 2),
        current_best_r_cell=round(y_best, 4),
        grid_x=[round(float(v), 3) for v in grid],
        gp_mean=[round(float(v), 5) for v in mu],
        gp_lower_95=[round(float(v), 5) for v in (mu - 1.96 * sigma)],
        gp_upper_95=[round(float(v), 5) for v in (mu + 1.96 * sigma)],
        ei_curve=[round(float(v), 6) for v in ei],
        observed_x=[round(float(v), 3) for v in x_obs],
        observed_r_cell=[round(float(v), 5) for v in y_obs],
        rationale=(
            f"Recommend next operando scan at {x_next:.1f} {parameter_unit} "
            f"(predicted R_cell = {pred_mean:.3f} ± {1.96 * pred_std:.3f}, "
            f"EI = {ei_val:.4f}; current optimum {x_best:.1f} {parameter_unit} with R_cell = {y_best:.3f})."
        ),
    )
