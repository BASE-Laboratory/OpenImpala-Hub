"""Microstructure-to-cell C-rate polarisation & galvanostatic discharge predictor (`Pillar B`).

Bridges homogenised `OpenImpala` microstructural transport metrics
`(porosity, tortuosity_z, d_eff_z, specific_surface_area_m_inv, solid_sigma_z, thickness_m)`
directly into cell-level electrochemical figures of merit across `0.5C–4C`:
- Electrolyte salt-depletion limiting current density `J_lim` [mA/cm²] and critical C-rate `C_lim`,
- Newman porous-electrode ohmic (`ΔV_ionic`, `ΔV_electronic`), Butler–Volmer kinetic (`ΔV_kinetic`),
  and transient Nernst–Planck concentration (`ΔV_conc`) overpotential waterfall,
- Simulated galvanostatic voltage–capacity curves `V(Q)` showing accessible capacity fade
  when high through-plane tortuosity triggers electrolyte depletion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import math
from typing import Any, Callable, Sequence

import numpy as np

FARADAY_C_PER_MOL = 96485.3321
GAS_R_J_MOL_K = 8.3144626
TEMP_K = 298.15
THERMAL_V = (GAS_R_J_MOL_K * TEMP_K) / FARADAY_C_PER_MOL  # ~0.025693 V


@dataclass(frozen=True)
class ChemistrySpec:
    """Electrolyte, interfacial, and open-circuit parameters for a cell chemistry."""

    key: str
    title: str
    v_cutoff_low: float
    v_nominal: float
    c0_mol_m3: float
    d0_m2_s: float
    kappa0_s_m: float
    sigma0_s_m: float
    t_plus: float
    i0_a_m2: float
    areal_capacity_mah_cm2: float
    min_realistic_thickness_um: float
    ocv_fn: Callable[[float], float]


def _ocv_nmc811(soc: float) -> float:
    s = min(max(float(soc), 0.005), 0.995)
    return (
        3.42
        + 0.76 * s
        - 0.12 * s * (1.0 - s)
        - 0.28 * math.exp(-20.0 * s)
        + 0.07 * math.exp(-16.0 * (1.0 - s))
    )


def _ocv_lfp(soc: float) -> float:
    s = min(max(float(soc), 0.005), 0.995)
    return (
        3.32
        + 0.045 * (s - 0.5)
        - 0.52 * math.exp(-26.0 * s)
        + 0.16 * math.exp(-22.0 * (1.0 - s))
    )


def _ocv_vanadium(soc: float) -> float:
    s = min(max(float(soc), 0.01), 0.99)
    return 1.39 + 2.0 * THERMAL_V * math.log(s / (1.0 - s))


CHEMISTRIES: dict[str, ChemistrySpec] = {
    "nmc811_li_ion": ChemistrySpec(
        key="nmc811_li_ion",
        title="NMC811 | Graphite Li-Ion (1 M LiPF6)",
        v_cutoff_low=3.00,
        v_nominal=3.82,
        c0_mol_m3=1000.0,
        d0_m2_s=2.6e-10,
        kappa0_s_m=1.05,
        sigma0_s_m=42.0,
        t_plus=0.38,
        i0_a_m2=1.5,
        areal_capacity_mah_cm2=3.5,
        min_realistic_thickness_um=65.0,
        ocv_fn=_ocv_nmc811,
    ),
    "lfp_high_rate": ChemistrySpec(
        key="lfp_high_rate",
        title="LFP | Graphite High-Rate (1.2 M LiPF6)",
        v_cutoff_low=2.50,
        v_nominal=3.30,
        c0_mol_m3=1200.0,
        d0_m2_s=3.2e-10,
        kappa0_s_m=1.18,
        sigma0_s_m=55.0,
        t_plus=0.38,
        i0_a_m2=2.1,
        areal_capacity_mah_cm2=2.8,
        min_realistic_thickness_um=55.0,
        ocv_fn=_ocv_lfp,
    ),
    "vanadium_flow_felt": ChemistrySpec(
        key="vanadium_flow_felt",
        title="Vanadium / Aqueous Redox Carbon Felt",
        v_cutoff_low=0.90,
        v_nominal=1.38,
        c0_mol_m3=1600.0,
        d0_m2_s=3.9e-10,
        kappa0_s_m=24.0,
        sigma0_s_m=380.0,
        t_plus=0.70,
        i0_a_m2=4.2,
        areal_capacity_mah_cm2=8.0,
        min_realistic_thickness_um=95.0,
        ocv_fn=_ocv_vanadium,
    ),
}


@dataclass(frozen=True)
class CRateCurve:
    """Simulated galvanostatic discharge curve & mid-SOC overpotential breakdown for one C-rate."""

    c_rate: float
    current_density_ma_cm2: float
    accessible_capacity_pct: float
    specific_energy_pct: float
    eta_ionic_mv: float
    eta_electronic_mv: float
    eta_kinetic_mv: float
    eta_concentration_mv: float
    capacity_pct_grid: list[float]
    voltage_v_grid: list[float]


@dataclass(frozen=True)
class ElectrochemReport:
    """Serializable cell-level electrochemical performance prediction from microstructure metrics."""

    chemistry_key: str
    chemistry_title: str
    effective_thickness_um: float
    limiting_current_ma_cm2: float
    limiting_c_rate: float
    diffusion_relaxation_s: float
    pybamm_available: bool
    curves: list[CRateCurve]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def has_pybamm_runtime() -> bool:
    return importlib.util.find_spec("pybamm") is not None


def evaluate_cell_electrochemistry(
    *,
    porosity: float,
    d_eff_z: float,
    specific_surface_area_m_inv: float,
    solid_sigma_eff_z: float,
    thickness_m: float,
    chemistry_key: str = "nmc811_li_ion",
    c_rates: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    preserve_exact_thickness: bool = False,
) -> ElectrochemReport:
    """Evaluate C-rate discharge curves and overpotential breakdown from homogenised metrics."""
    chem = CHEMISTRIES.get(chemistry_key, CHEMISTRIES["nmc811_li_ion"])

    if preserve_exact_thickness:
        l_eff_m = max(float(thickness_m), 5.0e-6)
    else:
        l_eff_m = min(
            max(float(thickness_m), chem.min_realistic_thickness_um * 1e-6),
            800.0e-6,
        )

    eps_eff = max(float(porosity), 0.02)
    deff_rel = max(float(d_eff_z), 1.0e-4)
    sig_rel = max(float(solid_sigma_eff_z), 0.012)
    a_s = max(float(specific_surface_area_m_inv), 2.5e4)

    d_eff_m2_s = chem.d0_m2_s * deff_rel
    kappa_eff = chem.kappa0_s_m * deff_rel
    sigma_eff = chem.sigma0_s_m * sig_rel

    # Salt-depletion limiting current density: J_lim = 3 F D_eff c_0 / ((1 - t_+) L) [A/m^2]
    j_lim_a_m2 = (3.0 * FARADAY_C_PER_MOL * d_eff_m2_s * chem.c0_mol_m3) / (
        max(1.0 - chem.t_plus, 0.05) * l_eff_m
    )
    # 1 mA/cm^2 == 10 A/m^2
    i_1c_a_m2 = chem.areal_capacity_mah_cm2 * 10.0
    j_lim_ma_cm2 = j_lim_a_m2 / 10.0
    c_lim = j_lim_a_m2 / max(i_1c_a_m2, 1e-9)

    tau_diff_s = (eps_eff * (l_eff_m**2)) / max((math.pi**2) * d_eff_m2_s, 1e-18)

    u_grid = np.linspace(0.0, 0.99, 75, dtype=np.float64)
    ideal_energy = float(
        np.trapezoid([chem.ocv_fn(1.0 - float(u)) for u in u_grid], u_grid)
    )

    curves: list[CRateCurve] = []
    for c_val in c_rates:
        c_rate = max(0.05, float(c_val))
        i_app = c_rate * i_1c_a_m2  # [A/m^2]

        # Newman porous-electrode ohmic & Butler-Volmer activation losses [V]
        dv_ion = (i_app * l_eff_m) / (3.0 * max(kappa_eff, 1e-6))
        dv_elec = (i_app * l_eff_m) / (3.0 * max(sigma_eff, 1e-6))
        j_bv = i_app / max(2.0 * l_eff_m * a_s * chem.i0_a_m2, 1e-6)
        dv_kin = 2.0 * THERMAL_V * math.asinh(j_bv)

        cap_pts: list[float] = []
        volt_pts: list[float] = []
        mid_dv_conc = 0.0

        for u in u_grid:
            t_s = (3600.0 * float(u)) / c_rate
            relax = 1.0 - math.exp(-t_s / max(tau_diff_s, 0.25))
            theta_dep = (i_app / max(j_lim_a_m2, 1e-6)) * relax

            if theta_dep >= 0.992:
                # Electrolyte salt concentration depleted to zero -> cutoff reached
                if cap_pts:
                    cap_pts.append(round(float(u) * 100.0, 2))
                    volt_pts.append(round(chem.v_cutoff_low, 4))
                break

            dv_conc = (
                2.0
                * THERMAL_V
                * (1.0 - chem.t_plus)
                * abs(math.log(max(1.0 - theta_dep, 0.008)))
            )
            if abs(float(u) - 0.50) < 0.02:
                mid_dv_conc = dv_conc

            v_term = chem.ocv_fn(1.0 - float(u)) - dv_ion - dv_elec - dv_kin - dv_conc
            if v_term <= chem.v_cutoff_low:
                cap_pts.append(round(float(u) * 100.0, 2))
                volt_pts.append(round(chem.v_cutoff_low, 4))
                break

            cap_pts.append(round(float(u) * 100.0, 2))
            volt_pts.append(round(v_term, 4))

        if not cap_pts:
            cap_pts = [0.0]
            volt_pts = [round(chem.v_cutoff_low, 4)]

        acc_pct = float(cap_pts[-1])
        if mid_dv_conc == 0.0:
            # High-rate cutoff occurred before 50% SOC: report concentration drop just prior to cutoff
            theta_end = min(0.985, i_app / max(j_lim_a_m2, 1e-6))
            mid_dv_conc = (
                2.0
                * THERMAL_V
                * (1.0 - chem.t_plus)
                * abs(math.log(max(1.0 - theta_end, 0.015)))
            )

        delivered_energy = (
            float(np.trapezoid(volt_pts, np.asarray(cap_pts) / 100.0))
            if len(cap_pts) >= 2
            else 0.0
        )
        energy_pct = min(100.0, 100.0 * delivered_energy / max(ideal_energy, 1e-6))

        curves.append(
            CRateCurve(
                c_rate=round(c_rate, 2),
                current_density_ma_cm2=round(i_app / 10.0, 3),
                accessible_capacity_pct=round(acc_pct, 2),
                specific_energy_pct=round(energy_pct, 2),
                eta_ionic_mv=round(dv_ion * 1000.0, 4),
                eta_electronic_mv=round(dv_elec * 1000.0, 4),
                eta_kinetic_mv=round(dv_kin * 1000.0, 4),
                eta_concentration_mv=round(mid_dv_conc * 1000.0, 4),
                capacity_pct_grid=cap_pts,
                voltage_v_grid=volt_pts,
            )
        )

    return ElectrochemReport(
        chemistry_key=chem.key,
        chemistry_title=chem.title,
        effective_thickness_um=round(l_eff_m * 1e6, 2),
        limiting_current_ma_cm2=round(j_lim_ma_cm2, 5),
        limiting_c_rate=round(c_lim, 5),
        diffusion_relaxation_s=round(tau_diff_s, 3),
        pybamm_available=has_pybamm_runtime(),
        curves=curves,
    )
