import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.pk_model import simulate_population, DOSING_REGIMENS
from src.parameter_estimation import estimate_all_subjects, covariate_adjusted_population_estimate, bootstrap_population_estimate


def test_at_least_800_subjects():
    subjects = simulate_population()
    assert len(subjects) >= 800


def test_6_dosing_regimens():
    assert len(DOSING_REGIMENS) == 6


def test_parameter_estimation_recovers_true_cl():
    subjects = simulate_population(n=60)
    estimated = estimate_all_subjects(subjects)
    errors = [abs(s["cl_est"] - s["true_cl"]) / s["true_cl"] for s in estimated]
    assert np.mean(errors) < 0.15


def test_population_estimate_recovers_weight_exponent():
    subjects = simulate_population(n=300)
    estimated = estimate_all_subjects(subjects)
    pop_est = covariate_adjusted_population_estimate(estimated)
    assert abs(pop_est["weight_exponent"] - 0.75) < 0.2


def test_bootstrap_ci_contains_point_estimate():
    subjects = simulate_population(n=150)
    estimated = estimate_all_subjects(subjects)
    ci = bootstrap_population_estimate(estimated, n_boot=50)
    for k, v in ci.items():
        assert v["ci_lower"] <= v["mean"] <= v["ci_upper"]
