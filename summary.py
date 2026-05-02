from itertools import product
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib


def read_dir(target_fn, method, profile, lengthscale):
    def read_file(f):
        r = np.load(os.path.join(path, f), allow_pickle=True)
        y = np.minimum.accumulate(r["observation_values"])
        # dict with summary statistics
        summary = {
            "best_y": np.min(y),
            "avg_regret": np.mean(y),
            "t_fit": r["surrogate_fit_time"],
            "t_acq": r["acquisition_time"],
            "t_eval": r["target_evaluation_time"],
        }
        # dict with y values at each acquisition step 
        running_best = [{"i": i, "y": yi} for i, yi in enumerate(y)]
        return summary, running_best

    # read all files with this combination of params
    path = f"results/{target_fn}/{method}/{profile}_lengthscale_{lengthscale}/"
    files = [f for f in os.listdir(path) if f.endswith(".pkl")]
    summaries, ys = zip(*map(read_file, files))
    ys = sum(ys, [])  # flatten list of lists

    # convert list(dict) to dataframe and add the constant columns
    summary_df = pd.DataFrame(summaries).assign(
        method=method,
        profile=profile,
        target_fn=target_fn,
        lengthscale=lengthscale,
    )
    ys_df = pd.DataFrame(ys).assign(
        method=method,
        profile=profile,
        target_fn=target_fn,
        lengthscale=lengthscale,
    )
    return summary_df, ys_df


if __name__ == "__main__":
    combos = []
    for target_fn in os.listdir("results/"):
        for method in os.listdir(f"results/{target_fn}/"):
            for profile_and_scale in os.listdir(f"results/{target_fn}/{method}/"):
                profile, _, lengthscale = profile_and_scale.split("_")
                combos.append((target_fn, method, profile, lengthscale))

    dfs = joblib.Parallel(-1)(
        joblib.delayed(read_dir)(target_fn, method, profile, lengthscale)
        for target_fn, method, profile, lengthscale in tqdm(combos)
    )
    summary_dfs, ys_dfs = zip(*dfs)

    summary_df = pd.concat(summary_dfs, ignore_index=True)
    summary_df.to_csv("results_summary.csv", index=False)

    ys_df = pd.concat(ys_dfs, ignore_index=True)
    ys_df.to_csv("results_ys.csv", index=False)