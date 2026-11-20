"""Real one-compartment population PK model with inter-individual
variability and covariate effects on clearance/volume (nonlinear
mixed-effects simulation)."""
from __future__ import annotations

import numpy as np

N_SUBJECTS = 820
SEED = 606

DOSING_REGIMENS = [
    {"name": "regimen_A_50mg_qd", "dose": 50, "interval": 24, "n_doses": 7},
    {"name": "regimen_B_100mg_qd", "dose": 100, "interval": 24, "n_doses": 7},
    {"name": "regimen_C_50mg_bid", "dose": 50, "interval": 12, "n_doses": 14},
    {"name": "regimen_D_200mg_qd", "dose": 200, "interval": 24, "n_doses": 7},
    {"name": "regimen_E_100mg_bid", "dose": 100, "interval": 12, "n_doses": 14},
    {"name": "regimen_F_single_150mg", "dose": 150, "interval": 24, "n_doses": 1},
]
assert len(DOSING_REGIMENS) == 6

# population (true) PK parameters
POP_CL = 5.0   # L/h, typical clearance
POP_V = 50.0   # L, typical volume of distribution
POP_KA = 1.0   # 1/h, absorption rate constant
OMEGA_CL = 0.3  # inter-individual variability (lognormal SD) on CL
OMEGA_V = 0.2


def one_compartment_oral_conc(t, dose, ka, ke, v):
    """Concentration at time t after a single oral dose (one-compartment,
    first-order absorption/elimination)."""
    if abs(ka - ke) < 1e-6:
        ke = ke + 1e-6
    return (dose * ka) / (v * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


def simulate_subject_concentration_profile(dose, interval, n_doses, cl, v, ka, t_eval):
    ke = cl / v
    conc = np.zeros_like(t_eval, dtype=float)
    for dose_idx in range(n_doses):
        dose_time = dose_idx * interval
        mask = t_eval >= dose_time
        conc[mask] += one_compartment_oral_conc(t_eval[mask] - dose_time, dose, ka, ke, v)
    return conc


def simulate_population(n: int = N_SUBJECTS, seed: int = SEED):
    rng = np.random.default_rng(seed)
    weight = rng.normal(75, 15, n).clip(40, 140)
    age = rng.normal(50, 15, n).clip(18, 90)

    # true covariate effects: weight affects both CL and V (allometric-style),
    # age affects CL (renal/hepatic function decline)
    cl_cov = POP_CL * (weight / 70) ** 0.75 * (1 - 0.005 * (age - 50))
    v_cov = POP_V * (weight / 70) ** 1.0

    eta_cl = rng.normal(0, OMEGA_CL, n)
    eta_v = rng.normal(0, OMEGA_V, n)
    cl_i = cl_cov * np.exp(eta_cl)
    v_i = v_cov * np.exp(eta_v)
    ka_i = np.full(n, POP_KA) * np.exp(rng.normal(0, 0.1, n))

    regimen_idx = rng.integers(0, len(DOSING_REGIMENS), n)

    subjects = []
    for i in range(n):
        regimen = DOSING_REGIMENS[regimen_idx[i]]
        t_end = regimen["interval"] * regimen["n_doses"] + 24
        t_eval = np.arange(0, t_end, 1.0)
        conc_true = simulate_subject_concentration_profile(
            regimen["dose"], regimen["interval"], regimen["n_doses"], cl_i[i], v_i[i], ka_i[i], t_eval
        )
        noise = rng.normal(0, 0.08 * np.maximum(conc_true, 0.01))
        conc_observed = np.maximum(conc_true + noise, 0)

        subjects.append({
            "subject_id": f"S{i:04d}", "weight": weight[i], "age": age[i],
            "regimen": regimen["name"], "true_cl": cl_i[i], "true_v": v_i[i], "true_ka": ka_i[i],
            "t_eval": t_eval, "conc_observed": conc_observed,
        })
    return subjects
