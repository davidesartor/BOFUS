import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_ys(ax, ys: np.ndarray, style: dict):
    runs, max_acquisitions = ys.shape
    y_best = np.minimum.accumulate(ys, axis=-1)
    x = np.arange(1, max_acquisitions + 1)
    qu = np.clip(0.5 + 1.96 * np.sqrt(0.25 / runs), 0, 1)
    ql = np.clip(0.5 - 1.96 * np.sqrt(0.25 / runs), 0, 1)
    ul = np.quantile(y_best, qu, axis=0)
    ll = np.quantile(y_best, ql, axis=0)
    ax.plot(x, np.median(y_best, axis=0), **style)
    ax.fill_between(x, ll, ul, alpha=0.2, color=ax.lines[-1].get_color())


def filter_and_rename_methods(
    df: pd.DataFrame, methods: dict[str, str]
) -> pd.DataFrame:
    # filters method column by dict and renames according to dict values
    dfs = []
    for method, new_method in methods.items():
        df_method = df[df["method"] == method]
        df_method["method"] = new_method
        dfs.append(df_method)
    return pd.concat(dfs, ignore_index=True)


def plot_running_best(df: pd.DataFrame, save_dir: str, methods: dict[str, str]):
    os.makedirs(save_dir, exist_ok=True)
    df = filter_and_rename_methods(df, methods)

    for target_fn in df["target_fn"].unique():
        df_targ = df[df["target_fn"] == target_fn]
        assert len(df_targ["profile"].unique()) == 1, "Expected only one profile"
        assert len(df_targ["lengthscale"].unique()) == 1, "Expected only one scale"
        profile = df_targ["profile"].iloc[0]
        lengthscale = df_targ["lengthscale"].iloc[0]

        fig = plt.figure(figsize=(8, 6))
        for method in df_targ["method"].unique():
            df_method = df_targ[df_targ["method"] == method]
            ys = [
                df_method[df_method["i"] == i]["y"].values
                for i in sorted(df_method["i"].unique())
            ]
            ys = np.stack(ys, axis=1)  # shape (runs, acquisitions)
            plot_ys(plt.gca(), ys, style={"label": method})

        plt.title(f"{target_fn} (profile={profile}, lengthscale={lengthscale})")
        plt.xlabel("Acquisitions")
        plt.ylabel("Best y")
        # scale escluding random initial acquisitions to avoid skewing the plot
        ys_excl_first_10 = df_targ[df_targ["i"] >= 10]["y"].values      
        y_min = ys_excl_first_10.min()
        y_max = ys_excl_first_10.max()
        plt.ylim(y_min * 0.9, y_max * 1.1)
        plt.yscale("log")
        plt.legend()
        plt.savefig(f"{save_dir}/{target_fn}.pdf")
        plt.close()


if __name__ == "__main__":
    # find the best combination of profile/lengthscale for each target function
    summary = pd.read_csv("results_summary.csv")
    # exclude vellanky since it is invariant to lengthscale
    summary = summary[summary["method"] != "vellanky"]
    best_combos = summary.loc[
        summary.groupby("target_fn")["best_y"].idxmin(),
        ["target_fn", "profile", "lengthscale"],
    ]

    # filter the ys to only include the best combinations
    ys = pd.read_csv("results_ys.csv")
    ys = ys.merge(best_combos, on=["target_fn", "profile", "lengthscale"])

    ################################################################################
    # NATURAL GRADIENT ABLATIONS
    for method in ["wycoff", "vien"]:
        plot_running_best(
            df=ys,
            save_dir=f"plots/natural_gradient_ablation_{method}",
            methods={
                f"{method}": f"natural gradient",
                f"{method}_no_natural_grad": f"vanilla gradient",
            },
        )

    ################################################################################
    # CANDIDATES SAMPLING ABLATIONS
    plot_running_best(
        df=ys,
        save_dir=f"plots/candidates_sampling_ablation",
        methods={
            f"wycoff": f"wycoff",
            f"wycoff_sample_from_gp": f"wycoff (sample from GP)",
            f"shilton": f"shilton",
        },
    )

    ################################################################################
    # METHOD COMPARISON
    plot_running_best(
        df=ys,
        save_dir="plots/method_comparison",
        methods={
            "wycoff": "wycoff",
            "wycoff_no_natural_grad": "wycoff (vanilla)",
            "kundu": "kundu",
            "vien": "vien",
            "vien_no_natural_grad": "vien (vanilla)",
            "shilton": "shilton",
            "vellanky": "vellanky",
        },
    )
