import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import joblib


def plot_times(ax, times: np.ndarray, position: int):
    ax.boxplot(times, positions=[position], patch_artist=True)
    ax.set_ylabel(f"Time (seconds)")
    ax.grid(True)


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


def plot(
    target_fn: str,
    lengthscale: float,
    methods: list[str],
    profiles: list[str],
    savepath: str,
):
    print(f"Plotting results for {target_fn} (lengthscale {lengthscale})")
    colors = plt.cm.tab10.colors
    linestyles = ["-", "--", ":", "-."]

    fig, (ax_ys, ax_leg) = plt.subplots(
        1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [3, 1]}
    )

    for i, (method, color) in enumerate(zip(methods, colors)):
        for profile, linestyle in zip(profiles, linestyles):
            save_dir = (
                f"results/{method}/{profile}/{target_fn}/lengthscale_{lengthscale}/"
            )
            try:
                results = [
                    np.load(os.path.join(save_dir, f), allow_pickle=True)
                    for f in os.listdir(save_dir)
                    if f.endswith(".pkl")
                ]
            except FileNotFoundError:
                print(f"Directory not found: {save_dir}")
                continue
            ys = np.array([r["observation_values"] for r in results])
            plot_ys(ax_ys, ys, style={"linestyle": linestyle, "color": color})

    ax_ys.set(title="Target function", ylabel="Target", xlabel="Acquisitions")
    ax_ys.grid(True)
    ax_ys.set_yscale("log")

    method_handles = [
        mlines.Line2D([], [], color=colors[i], linewidth=2, label=method)
        for i, method in enumerate(methods)
    ]
    profile_handles = [
        mlines.Line2D(
            [], [], color="black", linestyle=linestyles[i], linewidth=2, label=profile
        )
        for i, profile in enumerate(profiles)
    ]
    ax_leg.axis("off")
    ax_leg.legend(
        handles=method_handles
        + [mlines.Line2D([], [], color="none", label="")]
        + profile_handles,
        loc="center left",
        frameon=True,
        title="Method / Kernel",
    )

    fig.suptitle(f"{target_fn} ($\\rho={lengthscale}$)")
    fig.tight_layout()

    os.makedirs(f"plots/{target_fn}", exist_ok=True)
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)
    print(f"Done!\n")


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

    methods = ["wycoff", "kundu", "vien", "shilton", "vellanky", "random"]
    joblib.Parallel(n_jobs=-1)(
        joblib.delayed(plot)(
            target_fn,
            lengthscale,
            methods,
            profiles,
            f"plots/{target_fn}/lengthscale_{lengthscale}.pdf",
        )
        for lengthscale in lengthscales
        for target_fn in targets
    )

    methods = ["wycoff", "wycoff_no_natural_grad", "vien", "vien_no_natural_grad"]
    joblib.Parallel(n_jobs=-1)(
        joblib.delayed(plot)(
            target_fn,
            lengthscale,
            methods,
            profiles,
            f"plots/{target_fn}/lengthscale_{lengthscale}_gradient_ablation.pdf",
        )
        for lengthscale in lengthscales
        for target_fn in targets
    )

    methods = ["wycoff", "wycoff_sample_from_gp", "shilton"]
    joblib.Parallel(n_jobs=-1)(
        joblib.delayed(plot)(
            target_fn,
            lengthscale,
            methods,
            profiles,
            f"plots/{target_fn}/lengthscale_{lengthscale}_sampling_ablation.pdf",
        )
        for lengthscale in lengthscales
        for target_fn in targets
    )
