"""Real per-subject nonlinear least-squares parameter estimation
(scipy.optimize.curve_fit), covariate-adjusted population parameter
estimation, and bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from .pk_model import DOSING_REGIMENS, simulate_subject_concentration_profile


def _regimen_lookup():
    return {r["name"]: r for r in DOSING_REGIMENS}


_REGIMENS = _regimen_lookup()


def estimate_subject_params(subject: dict) -> dict | None:
    regimen = _REGIMENS[subject["regimen"]]

    def model(t, cl, v, ka):
        return simulate_subject_concentration_profile(
            regimen["dose"], regimen["interval"], regimen["n_doses"], cl, v, ka, t
        )

    try:
        popt, _ = curve_fit(
            model, subject["t_eval"], subject["conc_observed"],
            p0=[5.0, 50.0, 1.0], bounds=([0.5, 5.0, 0.1], [50.0, 300.0, 5.0]), maxfev=5000,
        )
        return {"cl_est": popt[0], "v_est": popt[1], "ka_est": popt[2]}
    except Exception:
        return None


def estimate_all_subjects(subjects: list) -> list:
    results = []
    for s in subjects:
        est = estimate_subject_params(s)
        if est is not None:
            results.append({**s, **est})
    return results


def covariate_adjusted_population_estimate(estimated_subjects: list) -> dict:
    """Fit CL_i = theta_CL * (weight/70)^0.75 * (1 - theta_age*(age-50))
    via log-linearized least squares to recover population-level
    covariate effects from the per-subject estimates."""
    weight = np.array([s["weight"] for s in estimated_subjects])
    age = np.array([s["age"] for s in estimated_subjects])
    cl_est = np.array([s["cl_est"] for s in estimated_subjects])

    log_cl = np.log(cl_est)
    log_weight_ratio = np.log(weight / 70)
    age_centered = age - 50

    X = np.column_stack([np.ones_like(log_cl), log_weight_ratio, age_centered])
    coefs, _, _, _ = np.linalg.lstsq(X, log_cl, rcond=None)
    theta_cl0 = np.exp(coefs[0])
    weight_exponent = coefs[1]
    age_effect = -coefs[2]

    return {"theta_cl_typical": float(theta_cl0), "weight_exponent": float(weight_exponent),
            "age_effect": float(age_effect)}


def bootstrap_population_estimate(estimated_subjects: list, n_boot: int = 200, seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    n = len(estimated_subjects)
    boot_results = {"theta_cl_typical": [], "weight_exponent": [], "age_effect": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = [estimated_subjects[i] for i in idx]
        est = covariate_adjusted_population_estimate(sample)
        for k in boot_results:
            boot_results[k].append(est[k])

    ci = {}
    for k, vals in boot_results.items():
        vals = np.array(vals)
        ci[k] = {"mean": round(float(vals.mean()), 4),
                  "ci_lower": round(float(np.percentile(vals, 2.5)), 4),
                  "ci_upper": round(float(np.percentile(vals, 97.5)), 4)}
    return ci
