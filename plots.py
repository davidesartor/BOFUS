import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import pickle


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
        fig = plt.figure(figsize=(4, 4))
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
        # scale escluding random initial acquisitions to avoid skewing the plot
        plt.yscale("log")
        plt.xlim(1, df_targ["i"].max() + 1)
        if target_fn == "mnist":
            plt.ylim(2.0e-2, 2.5e-2)
        if target_fn == "sinc3d":
            plt.ylim(2.0e-2, 1.1e-1)
        if target_fn == "sinc4d":
            plt.ylim(2.0e-2, 1.1e-1)
        plt.grid()
        plt.legend(loc="upper right")  # top right
        plt.savefig(f"{save_dir}/{target_fn}.pdf", bbox_inches="tight")
        plt.close()


def print_results_table(df: pd.DataFrame, save_dir: str, methods: dict[str, str]):
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
        return f"{median:.4g} [{lo:.4g}, {hi:.4g}]"

    for metric in metrics:
        rows = pd.DataFrame(
            {
                target_fn: [get_cell(m, target_fn, metric) for m in method_names]
                for target_fn in target_fns
            },
            index=method_names,
        )
        with open(f"{save_dir}/{metric}_table.txt", "w") as f:
            f.write(rows.to_string())


def plot_mnist_activations(df: pd.DataFrame, save_dir: str, methods: dict[str, str]):
    os.makedirs(save_dir, exist_ok=True)
    df = df[df["target_fn"] == "mnist"]
    method_colors = {m: f"C{i}" for i, m in enumerate(methods.keys())}

    x = jnp.linspace(-3, 3, 1000)
    plt.figure(figsize=(8, 4))

    for method, label in tqdm(methods.items()):
        color = method_colors[method]
        df_method = df[df["method"] == method][
            ["profile", "lengthscale", "seed", "best_y"]
        ].sort_values("best_y")
        ys = []
        for _, row in df_method.iterrows():
            path = f"results/mnist/{method}/{row['profile']}_lengthscale_{row['lengthscale']}/seed_{row['seed']}.pkl"
            res = pickle.load(open(path, "rb"))
            f = res["observation_locations"][res["observation_values"].argmin()]
            activation = jax.vmap(lambda x: f((x[None] + 1.0) / 2.0) + jax.nn.relu(x))
            y = activation(x)
            ys.append(y)
            plt.plot(x, y, color=color, alpha=0.2)

        plt.plot(x, jnp.stack(ys).mean(axis=0), color=color, linewidth=3, label=label)

    plt.plot(x, jax.nn.relu(x), "k:", label="ReLU", linewidth=2)
    plt.xlim(-3, 3)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.legend()
    plt.savefig(f"{save_dir}/mnist_activations.pdf", bbox_inches="tight")
    plt.close()


def plot_brachistochrone_path(df: pd.DataFrame, save_dir: str, methods: dict[str, str]):
    os.makedirs(save_dir, exist_ok=True)
    df = df[df["target_fn"] == "brachistochrone"]
    from src.targets import Brachistochrone

    targ = Brachistochrone()
    method_colors = {m: f"C{i}" for i, m in enumerate(methods.keys())}

    x0, y0 = targ.initial_position
    x1, y1 = targ.final_position
    x = jnp.linspace(x0, x1, 1000)

    plt.figure(figsize=(8, 4))
    cycloid, optimal_time = targ.find_brachistochrone()

    for method, label in tqdm(methods.items()):
        color = method_colors[method]
        df_method = df[df["method"] == method][
            ["profile", "lengthscale", "seed", "best_y"]
        ].sort_values("best_y")
        ys = []
        for _, row in df_method.iterrows():
            path = f"results/brachistochrone/{method}/{row['profile']}_lengthscale_{row['lengthscale']}/seed_{row['seed']}.pkl"
            res = pickle.load(open(path, "rb"))
            f = res["observation_locations"][res["observation_values"].argmin()]
            t = targ(f)
            curve = jax.vmap(targ.get_curve(f))
            y = curve(x)
            ys.append(y)
            plt.plot(x, y, color=color, alpha=0.2)

        plt.plot(x, jnp.stack(ys).mean(axis=0), color=color, linewidth=3, label=label)

    plt.plot(x, cycloid(x), "k:", label=f"Cycloid", linewidth=2)
    plt.plot([x0, x1], [y0, y1], "ro")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()
    plt.savefig(f"{save_dir}/brachistochrone_path.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    ##############################################################################
    # load data
    print("Loading data...")
    summary_all = pd.read_csv("results/summary_all.csv")
    summary_filtered = pd.read_csv("results/summary_filtered.csv")
    ys_all = pd.read_csv("results/ys_all.csv")
    ys_filtered = pd.read_csv("results/ys_filtered.csv")

    # for each fn, get the best lengthscale/profile combination for each method
    # in summary filtered this is the unique combination left in the df
    print("Printing tables...")
    os.makedirs("plots/tables", exist_ok=True)
    best_combos = (
        summary_filtered.groupby(["target_fn", "method"])[["lengthscale", "profile"]]
        .first()
        .reset_index()
    )
    best_combos = filter_and_rename_methods(
        best_combos,
        methods={
            "wycoff_no_natural_grad": "ours",
            "kundu": "kundu",
            "vien": "vien",
            "shilton": "shilton",
        },
    )
    with open("plots/tables/best_lengthscales.txt", "w") as f:
        for target_fn in best_combos["target_fn"].unique():
            f.write(f"{target_fn}:\n")
            for method in best_combos[
                best_combos["target_fn"] == target_fn
            ].itertuples():
                f.write(
                    f"  {method.method}: lengthscale={method.lengthscale}, profile={method.profile}\n"
                )

    # get only the best lengthscale for each target_fn
    best_lengthscales = summary_filtered.groupby(["target_fn"])["lengthscale"].first()

    ################################################################################
    # METHOD COMPARISON
    print("Plotting method comparison...")
    plot_running_best(
        df=ys_filtered,
        title=f"",
        save_dir="plots/method_comparison",
        methods={
            "wycoff_no_natural_grad": "ours",
            "kundu": "kundu",
            "vien": "vien",
            "shilton": "shilton",
            "vellanky": "vellanky",
        },
    )

    ################################################################################
    # NATURAL GRADIENT ABLATIONS
    print("Plotting natural gradient ablations...")
    for method in ["wycoff", "vien"]:
        plot_running_best(
            df=ys_filtered,
            title=f"",
            save_dir=f"plots/natural_gradient_ablation_{method}",
            methods={
                f"{method}": f"Natural Gradient",
                f"{method}_no_natural_grad": f"Euclidean Gradient",
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
            f"wycoff": f"Ours",
            f"wycoff_sample_from_gp": f"Ours (sample from GP)",
            f"shilton": f"Shilton",
        },
    )

    ##############################################################################
    # KERNEL PROFILE ABLATIONS
    print("Plotting kernel profile ablations...")
    df = ys_all.copy()
    df["method"] = df["method"] + "_" + df["profile"]
    df = df.merge(best_lengthscales.reset_index(), on=["target_fn", "lengthscale"])
    plot_running_best(
        df=df,
        title=f"",
        save_dir=f"plots/kernel_profile_ablation",
        methods={
            f"wycoff_no_natural_grad_{profile}": f"{profile}"
            for profile in ys_all["profile"].unique()
        },
    )

    ###############################################################################
    # TABLES AVGREGRET AND BEST_Y
    print("Printing tables...")
    print_results_table(
        df=summary_filtered,
        save_dir="plots/tables",
        methods={
            "wycoff_no_natural_grad": "Ours",
            "kundu": "Kundu",
            "vien": "Vien",
            "shilton": "Shilton",
            "vellanky": "Vellanky",
        },
    )

    ##############################################################################
    # MNIST LEARNED ACTIVATION
    print("Plotting mnist learned activations...")
    plot_mnist_activations(
        df=summary_filtered,
        save_dir="plots/f_visualizations",
        methods={
            "wycoff_no_natural_grad": "Learned Activation",
            # "kundu": "kundu",
            # "vien": "vien",
            # "shilton": "shilton",
            # "vellanky": "vellanky",
        },
    )

    ##############################################################################
    # BRACHISTOCHRONE LEARNED PATH
    print("Plotting brachistochrone learned path...")
    plot_brachistochrone_path(
        df=summary_filtered,
        save_dir="plots/f_visualizations",
        methods={
            "wycoff_no_natural_grad": "Learned Path",
            # "kundu": "kundu",
            # "vien": "vien",
            # "shilton": "shilton",
            # "vellanky": "vellanky",
        },
    )
