import json
import os

import numpy as np

from src.pk_model import simulate_population, N_SUBJECTS
from src.parameter_estimation import (
    estimate_all_subjects, covariate_adjusted_population_estimate, bootstrap_population_estimate,
)


def main():
    subjects = simulate_population()
    estimated = estimate_all_subjects(subjects)

    cl_errors = [abs(s["cl_est"] - s["true_cl"]) / s["true_cl"] for s in estimated]
    v_errors = [abs(s["v_est"] - s["true_v"]) / s["true_v"] for s in estimated]

    pop_est = covariate_adjusted_population_estimate(estimated)
    boot_ci = bootstrap_population_estimate(estimated, n_boot=200)

    summary = {
        "n_subjects_simulated": N_SUBJECTS,
        "n_subjects_estimated": len(estimated),
        "mean_abs_pct_error_cl": round(100 * float(np.mean(cl_errors)), 2),
        "mean_abs_pct_error_v": round(100 * float(np.mean(v_errors)), 2),
        "population_estimate": {k: round(v, 4) for k, v in pop_est.items()},
        "bootstrap_ci": boot_ci,
        "true_weight_exponent": 0.75,
        "true_age_effect": 0.005,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
