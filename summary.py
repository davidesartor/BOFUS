from itertools import product
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib


def filter_best_lengthscale(df: pd.DataFrame) -> pd.DataFrame:
    # best lengthscale per target_fn across all (method, profile)
    best_lengthscale = (
        df.groupby(["target_fn", "method", "profile", "lengthscale"])["best_y"]
        .median()
        .reset_index()
        .loc[lambda d: d.groupby("target_fn")["best_y"].idxmin()][
            ["target_fn", "lengthscale"]
        ]
    )
    return df.merge(best_lengthscale, on=["target_fn", "lengthscale"])


def filter_best_profile(df: pd.DataFrame) -> pd.DataFrame:
    # best profile per (target_fn, method)
    best_profile = (
        df.groupby(["target_fn", "method", "profile"])["best_y"]
        .median()
        .reset_index()
        .loc[lambda d: d.groupby(["target_fn", "method"])["best_y"].idxmin()][
            ["target_fn", "method", "profile"]
        ]
    )
    return df.merge(best_profile, on=["target_fn", "method", "profile"])


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
    summary_df.to_csv("results_summary_all.csv", index=False)

    ys_df = pd.concat(ys_dfs, ignore_index=True)
    ys_df.to_csv("results_ys_all.csv", index=False)

    # filter to only include best profile and lengthscale for each target_fn and method
    summary_df = filter_best_lengthscale(summary_df)
    summary_df = filter_best_profile(summary_df)
    summary_df.to_csv("results_summary_filtered.csv", index=False)

    ys_df = ys_df.merge(
        summary_df[["target_fn", "method", "profile", "lengthscale"]].drop_duplicates(),
        on=["target_fn", "method", "profile", "lengthscale"],
    )
    ys_df.to_csv("results_ys_filtered.csv", index=False)
