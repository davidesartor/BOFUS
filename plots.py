import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import summary


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


def median_and_ci(data: np.ndarray, axis: int = 0):
    median = np.median(data, axis=axis)
    qu = np.clip(0.5 + 1.96 * np.sqrt(0.25 / data.shape[axis]), 0, 1)
    ql = np.clip(0.5 - 1.96 * np.sqrt(0.25 / data.shape[axis]), 0, 1)
    ci_upper = np.quantile(data, qu, axis=axis)
    ci_lower = np.quantile(data, ql, axis=axis)
    return median, ci_lower, ci_upper


def plot_ys(ax, ys: np.ndarray, style: dict):
    runs, max_acquisitions = ys.shape
    y_best = np.minimum.accumulate(ys, axis=-1)
    x = np.arange(1, max_acquisitions + 1)
    m, ll, ul = median_and_ci(y_best, axis=0)
    ax.plot(x, m, **style)
    ax.fill_between(x, ll, ul, alpha=0.2, color=ax.lines[-1].get_color())


def plot_running_best(
    df: pd.DataFrame, title: str, save_dir: str, methods: dict[str, str]
):
    os.makedirs(save_dir, exist_ok=True)
    df = filter_and_rename_methods(df, methods)

    for target_fn in tqdm(df["target_fn"].unique()):
        df_targ = df[df["target_fn"] == target_fn]
        fig = plt.figure(figsize=(8, 6))
        for method, color in zip(df_targ["method"].unique(), plt.cm.tab10.colors):
            df_method = df_targ[(df_targ["method"] == method)]
            ys = [
                df_method[df_method["i"] == i]["y"].values
                for i in sorted(df_method["i"].unique())
            ]
            ys = np.stack(ys, axis=1)  # shape (runs, acquisitions)
            plot_ys(plt.gca(), ys, style={"color": color, "label": method})

        plt.title(f"{target_fn}{title}")
        plt.xlabel("Acquisitions")
        plt.ylabel("Best y")
        # scale escluding random initial acquisitions to avoid skewing the plot
        plt.yscale("log")
        plt.xlim(1, df_targ["i"].max() + 1)
        plt.grid()
        plt.legend()
        plt.savefig(f"{save_dir}/{target_fn}.pdf")
        plt.close()


def print_results_table(df: pd.DataFrame, save_dir: str, methods: dict[str, str]):
    os.makedirs(save_dir, exist_ok=True)
    df = filter_and_rename_methods(df, methods)

    metrics = ["best_y", "avg_regret"]
    target_fns = sorted(df["target_fn"].unique())
    method_names = df["method"].unique().tolist()

    def get_cell(method, target_fn, metric):
        vals = df[(df["method"] == method) & (df["target_fn"] == target_fn)][
            metric
        ].values
        if len(vals) == 0:
            return "N/A"
        median, lo, hi = median_and_ci(vals)
        return f"{median:.3g} [{lo:.3g}, {hi:.3g}]"

    rows = pd.DataFrame(
        {
            (metric, target_fn): [get_cell(m, target_fn, metric) for m in method_names]
            for metric in metrics
            for target_fn in target_fns
        },
        index=method_names,
    )
    rows.columns = pd.MultiIndex.from_tuples(rows.columns)

    table_str = rows.to_string()
    print(table_str)
    with open(f"{save_dir}/results_table.txt", "w") as f:
        f.write(table_str)


if __name__ == "__main__":
    ##############################################################################
    # load data
    print("Loading data...")
    summary_all = pd.read_csv("results_summary_all.csv")
    summary_filtered = pd.read_csv("results_summary_filtered.csv")
    ys_all = pd.read_csv("results_ys_all.csv")
    ys_filtered = pd.read_csv("results_ys_filtered.csv")

    best_lengthscales = summary_filtered.groupby("target_fn")["lengthscale"].first()

    ################################################################################
    # NATURAL GRADIENT ABLATIONS
    print("Plotting natural gradient ablations...")
    for method in ["wycoff", "vien"]:
        plot_running_best(
            df=ys_filtered,
            title=f"",
            save_dir=f"plots/natural_gradient_ablation_{method}",
            methods={
                f"{method}": f"natural gradient",
                f"{method}_no_natural_grad": f"vanilla gradient",
            },
        )

    ################################################################################
    # CANDIDATES SAMPLING ABLATIONS
    print("Plotting candidates sampling ablations...")
    plot_running_best(
        df=ys_filtered,
        title=f"",
        save_dir=f"plots/candidates_sampling_ablation",
        methods={
            f"wycoff": f"wycoff",
            f"wycoff_sample_from_gp": f"wycoff (sample from GP)",
            f"shilton": f"shilton",
        },
    )

    ################################################################################
    # METHOD COMPARISON
    print("Plotting method comparison...")
    plot_running_best(
        df=ys_filtered,
        title=f"",
        save_dir="plots/method_comparison",
        methods={
            "wycoff_no_natural_grad": "wycoff",
            "kundu": "kundu",
            "vien": "vien",
            "shilton": "shilton",
            "vellanky": "vellanky",
        },
    )

    ##############################################################################
    # KERNEL PROFILE ABLATIONS
    print("Plotting kernel profile ablations...")
    plot_running_best(
        df=ys_all.merge(
            best_lengthscales.reset_index(), on=["target_fn", "lengthscale"]
        ),
        title=f"",
        save_dir=f"plots/kernel_profile_ablation",
        methods={
            f"wycoff_no_natural_grad_{profile}": f"wycoff {profile}"
            for profile in ys_filtered["profile"].unique()
        },
    )

    ###############################################################################
    # TABLE AVGREGRET
    print("Printing tables...")
    print("Best lengthscales:")
    print(best_lengthscales)
    with open("plots/tables/best_lengthscales.txt", "w") as f:
        f.write(" | ".join(f"{t}" for t in best_lengthscales.index))
        f.write("\n")
        f.write(" | ".join(f"{v}" for v in best_lengthscales.values))

    print_results_table(
        df=summary_filtered,
        save_dir="plots/tables",
        methods={
            "wycoff_no_natural_grad": "wycoff",
            "kundu": "kundu",
            "vien": "vien",
            "shilton": "shilton",
            "vellanky": "vellanky",
        },
    )
