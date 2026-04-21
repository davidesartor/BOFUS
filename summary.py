from itertools import product
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def read_dir(method, profile, target_fn, lengthscale):
    try:
        path = f"results/{method}/{profile}/{target_fn}/lengthscale_{lengthscale}/"
        rows = []
        for f in [f for f in os.listdir(path) if f.endswith(".pkl")]:
            r = np.load(os.path.join(path, f), allow_pickle=True)
            y = np.minimum.accumulate(r["observation_values"])
            best_y = np.min(y)
            rows.append(
                {
                    "method": method,
                    "profile": profile,
                    "target_fn": target_fn,
                    "lengthscale": lengthscale,
                    "best_y": best_y,
                    "avg_regret": np.mean(y),
                    "t_fit": r["surrogate_fit_time"],
                    "t_acq": r["acquisition_time"],
                    "t_eval": r["target_evaluation_time"],
                }
            )
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Error loading {method} {target_fn} l={lengthscale}: {e}")
        return None


if __name__ == "__main__":
    profiles = ["rbf", "matern52", "matern32"]
    targets = [
        "sinc",
        "gramacylee",
        "rosenbrock",
        "ackley",
        "hartmann",
        "pendulum",
        "mnist",
        "pinwheel",
    ]
    lengthscales = [0.3, 0.1, 0.03]
    methods = ["wycoff", "kundu", "vien", "shilton", "vellanky"]
    methods += [
        "wycoff_no_natural_grad",
        "vien_no_natural_grad",
        "wycoff_sample_from_gp",
    ]

    dfs = [
        read_dir(method, profile, target_fn, lengthscale)
        for method, profile, target_fn, lengthscale in tqdm(
            list(product(methods, profiles, targets, lengthscales))
        )
    ]
    dfs = [df for df in dfs if df is not None]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv("results.csv", index=False)
