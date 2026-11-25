r"""3D morphological, topological, and spectral validation suite (`Pillar A`).

Computes four complementary microstructural fidelity diagnostics on any calibrated
`basecoat.Volume`:
1. **6-Connected Pore-Network Partition**: Splits `conducting_phase` into
   *Percolating* (touches both inlet & outlet faces along `flow_direction`),
   *Dead-End* (touches one boundary face only), and *Trapped / Isolated*
   (disconnected interior cavities), satisfying
   `percolating + dead_end + trapped == bulk_porosity` identically.
2. **Inscribed Pore / Throat Diameter Distribution (`D10`, `D50`, `D90`)**:
   Combines `openimpala.particle_size_distribution` with 3D physical Euclidean
   distance-transform medial-axis radii (`2 * r_EDT` in µm) for continuous networks.
3. **3D Two-Point Correlation $S_2(r)$**: Exact $O(N^3 \log N)$ Wiener–Khinchin
   FFT autocorrelation along Z, in-plane XY, and isotropic radial shells,
   satisfying $S_2(0) = \varepsilon$ and $S_2(\infty) \to \varepsilon^2$.
4. **Directional Chord-Length Anisotropy**: Mean run-length chords `(Lz, Ly, Lx)` [µm].
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import ndimage
from basecoat import Volume
import openimpala as oi

from ..contracts import DirectionName


@dataclass(frozen=True)
class MorphologyReport:
    """Quantitative 3D morphological & connectivity diagnostics for a phase volume."""

    percolating_fraction: float
    dead_end_fraction: float
    trapped_fraction: float
    num_percolating_clusters: int
    num_dead_end_clusters: int
    num_trapped_clusters: int
    d10_um: float
    d50_um: float
    d90_um: float
    mean_diameter_um: float
    psd_bin_centres_um: list[float]
    psd_fractions: list[float]
    s2_r_um: list[float]
    s2_radial: list[float]
    s2_z_um: list[float]
    s2_z: list[float]
    s2_xy_um: list[float]
    s2_xy: list[float]
    mean_chord_um_zyx: tuple[float, float, float]
    chord_anisotropy_xy_over_z: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean_chord_1d(mask_3d: np.ndarray, axis: int, spacing_um: float) -> float:
    """Compute mean run-length chord along `axis` in micrometres."""
    moved = np.moveaxis(mask_3d, axis, -1)
    lines = moved.reshape(-1, moved.shape[-1]).astype(np.int8)
    n_lines, length = lines.shape
    if length == 0:
        return 0.0
    padded = np.zeros((n_lines, length + 2), dtype=np.int8)
    padded[:, 1:-1] = lines
    diffs = np.diff(padded, axis=1)
    _, starts = np.nonzero(diffs == 1)
    _, ends = np.nonzero(diffs == -1)
    if starts.size == 0:
        return 0.0
    chords = (ends - starts).astype(np.float64) * float(spacing_um)
    interior = (starts > 0) & (ends < length)
    if int(np.count_nonzero(interior)) >= 8:
        return float(np.mean(chords[interior]))
    return float(np.mean(chords))


def _partition_pore_connectivity(
    mask: np.ndarray,
    flow_direction: DirectionName,
) -> tuple[float, float, float, int, int, int]:
    """Partition 6-connected `mask` clusters into percolating, dead-end, and trapped fractions."""
    total_vox = float(mask.size)
    if total_vox == 0 or not np.any(mask):
        return 0.0, 0.0, 0.0, 0, 0, 0
    if bool(np.all(mask)):
        return 1.0, 0.0, 0.0, 1, 0, 0

    # 3D 6-connectivity matches 7-point finite-volume face transport
    struct6 = ndimage.generate_binary_structure(3, 1)
    labels, n_comp = ndimage.label(mask, structure=struct6)
    counts = np.bincount(labels.ravel())

    flow_ax = {"z": 0, "y": 1, "x": 2}[flow_direction]
    sl_in = [slice(None), slice(None), slice(None)]
    sl_out = [slice(None), slice(None), slice(None)]
    sl_in[flow_ax] = 0
    sl_out[flow_ax] = mask.shape[flow_ax] - 1

    inlet_ids = {int(v) for v in np.unique(labels[tuple(sl_in)]) if v > 0}
    outlet_ids = {int(v) for v in np.unique(labels[tuple(sl_out)]) if v > 0}

    perc_ids = inlet_ids & outlet_ids
    dead_ids = inlet_ids ^ outlet_ids
    all_ids = set(range(1, int(n_comp) + 1))
    trap_ids = all_ids - (inlet_ids | outlet_ids)

    perc_vox = sum(int(counts[i]) for i in perc_ids)
    dead_vox = sum(int(counts[i]) for i in dead_ids)
    trap_vox = sum(int(counts[i]) for i in trap_ids)

    return (
        round(perc_vox / total_vox, 6),
        round(dead_vox / total_vox, 6),
        round(trap_vox / total_vox, 6),
        len(perc_ids),
        len(dead_ids),
        len(trap_ids),
    )


def _compute_inscribed_psd(
    mask: np.ndarray,
    spacing_um_zyx: tuple[float, float, float],
    *,
    use_openimpala_psd: bool = True,
) -> tuple[float, float, float, float, list[float], list[float]]:
    """Compute inscribed pore/throat diameter quantiles `(D10, D50, D90)` and histogram in µm."""
    dz_um, dy_um, dx_um = spacing_um_zyx
    if not np.any(mask):
        return 0.0, 0.0, 0.0, 0.0, [0.0], [0.0]
    if bool(np.all(mask)):
        box_min = float(
            min(
                mask.shape[0] * dz_um,
                mask.shape[1] * dy_um,
                mask.shape[2] * dx_um,
            )
        )
        return box_min, box_min, box_min, box_min, [round(box_min, 3)], [1.0]

    diameters_um: np.ndarray | None = None
    if use_openimpala_psd:
        try:
            psd_res = oi.particle_size_distribution(
                mask.astype(np.int32),
                phase=1,
                voxel_size=(dz_um * 1e-6, dy_um * 1e-6, dx_um * 1e-6),
            )
            if psd_res.radii is not None and len(psd_res.radii) >= 12:
                diameters_um = 2.0 * np.asarray(psd_res.radii, dtype=np.float64) * 1e6
        except Exception:
            diameters_um = None

    if diameters_um is None or diameters_um.size == 0:
        # Continuous percolating network: extract 3D local medial-axis inscribed diameters 2 * EDT
        edt_um = ndimage.distance_transform_edt(mask, sampling=(dz_um, dy_um, dx_um))
        local_max = (edt_um == ndimage.maximum_filter(edt_um, size=3)) & mask & (edt_um > 0.0)
        radii_samples = edt_um[local_max]
        if radii_samples.size < 8:
            radii_samples = edt_um[mask]
        diameters_um = 2.0 * np.asarray(radii_samples, dtype=np.float64)

    d10, d50, d90 = (float(x) for x in np.percentile(diameters_um, [10.0, 50.0, 90.0]))
    d_mean = float(np.mean(diameters_um))

    d_lo, d_hi = float(np.min(diameters_um)), float(np.max(diameters_um))
    if d_hi <= d_lo + 1e-9:
        centres = [round(d_mean, 4)]
        fracs = [1.0]
    else:
        counts, edges = np.histogram(diameters_um, bins=16, range=(d_lo, d_hi))
        total = max(1, int(np.sum(counts)))
        centres = [round(float(0.5 * (edges[i] + edges[i + 1])), 4) for i in range(len(counts))]
        fracs = [round(float(c) / float(total), 5) for c in counts]

    return (
        round(d10, 4),
        round(d50, 4),
        round(d90, 4),
        round(d_mean, 4),
        centres,
        fracs,
    )


def _compute_two_point_s2(
    mask: np.ndarray,
    spacing_um_zyx: tuple[float, float, float],
    *,
    max_bins: int = 24,
) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
    r"""Compute exact periodic 3D FFT two-point correlation $S_2(r)$ satisfying $S_2(0) = \varepsilon$."""
    ind = mask.astype(np.float64)
    nz, ny, nx = ind.shape
    dz_um, dy_um, dx_um = spacing_um_zyx

    spec = np.fft.rfftn(ind, axes=(0, 1, 2))
    s2_3d = np.fft.irfftn(spec * np.conj(spec), s=ind.shape, axes=(0, 1, 2)) / float(ind.size)

    max_kz = max(2, min(nz // 2, max_bins))
    max_kxy = max(2, min(min(ny, nx) // 2, max_bins))

    s2_z_um = [round(k * dz_um, 4) for k in range(max_kz)]
    s2_z = [round(float(s2_3d[k, 0, 0]), 6) for k in range(max_kz)]

    dxy_um = 0.5 * (dy_um + dx_um)
    s2_xy_um = [round(k * dxy_um, 4) for k in range(max_kxy)]
    s2_xy = [
        round(0.5 * (float(s2_3d[0, k, 0]) + float(s2_3d[0, 0, k])), 6)
        for k in range(max_kxy)
    ]

    # Isotropic radial shell average over positive octant lags [0..max_lag]
    lz = min(nz // 2, max_bins)
    ly = min(ny // 2, max_bins)
    lx = min(nx // 2, max_bins)
    sub_s2 = s2_3d[:lz, :ly, :lx]
    zz, yy, xx = np.meshgrid(
        np.arange(lz, dtype=np.float64) * dz_um,
        np.arange(ly, dtype=np.float64) * dy_um,
        np.arange(lx, dtype=np.float64) * dx_um,
        indexing="ij",
    )
    r_grid = np.sqrt(zz * zz + yy * yy + xx * xx)
    r_max = float(
        min(
            (lz - 1) * dz_um if lz > 1 else dz_um,
            (ly - 1) * dy_um if ly > 1 else dy_um,
            (lx - 1) * dx_um if lx > 1 else dx_um,
        )
    )
    n_rad = max(4, min(max_bins, min(lz, ly, lx)))
    dr = max(r_max / float(max(1, n_rad - 1)), 1e-6)

    s2_r_um: list[float] = [0.0]
    s2_radial: list[float] = [round(float(s2_3d[0, 0, 0]), 6)]
    for b in range(1, n_rad):
        lo = (b - 0.5) * dr
        hi = (b + 0.5) * dr
        shell = (r_grid >= lo) & (r_grid < hi)
        if np.any(shell):
            s2_r_um.append(round(float(np.mean(r_grid[shell])), 4))
            s2_radial.append(round(float(np.mean(sub_s2[shell])), 6))

    return s2_r_um, s2_radial, s2_z_um, s2_z, s2_xy_um, s2_xy


def compute_morphology_report(
    phase_vol: Volume,
    *,
    conducting_phase: int = 0,
    flow_direction: DirectionName = "z",
    use_openimpala_psd: bool = True,
) -> MorphologyReport:
    """Compute full 3D connectivity, pore-size, chord, and $S_2(r)$ diagnostics on `phase_vol`."""
    arr = np.asarray(phase_vol.data)
    dz_um, dy_um, dx_um = phase_vol.spacing_in("um")
    mask = arr == int(conducting_phase)

    perc_f, dead_f, trap_f, n_perc, n_dead, n_trap = _partition_pore_connectivity(
        mask, flow_direction
    )
    d10, d50, d90, d_mean, psd_c, psd_f = _compute_inscribed_psd(
        mask,
        (dz_um, dy_um, dx_um),
        use_openimpala_psd=use_openimpala_psd,
    )
    s2_r_um, s2_rad, s2_z_um, s2_z, s2_xy_um, s2_xy = _compute_two_point_s2(
        mask,
        (dz_um, dy_um, dx_um),
    )

    lz_um = round(_mean_chord_1d(mask, 0, dz_um), 4)
    ly_um = round(_mean_chord_1d(mask, 1, dy_um), 4)
    lx_um = round(_mean_chord_1d(mask, 2, dx_um), 4)
    aniso_chord = (
        round(0.5 * (lx_um + ly_um) / lz_um, 4) if lz_um > 1e-9 else None
    )

    return MorphologyReport(
        percolating_fraction=perc_f,
        dead_end_fraction=dead_f,
        trapped_fraction=trap_f,
        num_percolating_clusters=n_perc,
        num_dead_end_clusters=n_dead,
        num_trapped_clusters=n_trap,
        d10_um=d10,
        d50_um=d50,
        d90_um=d90,
        mean_diameter_um=d_mean,
        psd_bin_centres_um=psd_c,
        psd_fractions=psd_f,
        s2_r_um=s2_r_um,
        s2_radial=s2_rad,
        s2_z_um=s2_z_um,
        s2_z=s2_z,
        s2_xy_um=s2_xy_um,
        s2_xy=s2_xy,
        mean_chord_um_zyx=(lz_um, ly_um, lx_um),
        chord_anisotropy_xy_over_z=aniso_chord,
    )
