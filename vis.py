import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
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


def plot_times(ax, times: np.ndarray, position: int):
    ax.boxplot(times, positions=[position], patch_artist=True)
    ax.set_ylabel(f"Time (seconds)")
    ax.grid(True)


def plot(target_fn: str, lengthscale: float, methods: list[str], profiles: list[str]):
    colors = plt.cm.tab10.colors
    linestyles = ["-", "--", ":", "-."]

    # 4-column grid: [ys (1/4) | fit (1/4) | acq (1/4) | legend (1/4)]
    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, 8, figure=fig)

    ax_ys = fig.add_subplot(gs[0, :3])
    ax_leg = fig.add_subplot(gs[:, 3])
    ax_fit = fig.add_subplot(gs[0, 4:6])
    ax_acq = fig.add_subplot(gs[0, 6:8])

    for i, (method, color) in enumerate(zip(methods, colors)):
        for profile, linestyle in zip(profiles, linestyles):
            save_dir = (
                f"results/{method}_{profile}/{target_fn}/lengthscale_{lengthscale}/"
            )
            try:
                results = [
                    np.load(os.path.join(save_dir, f), allow_pickle=True)
                    for f in os.listdir(save_dir)
                    if f.endswith(".pkl")
                ]
                assert len(results) == 16, len(results)
                ys = np.array([r["observation_values"] for r in results])
                t_fit = np.array([r["surrogate_fit_time"] for r in results])
                t_acq = np.array([r["acquisition_time"] for r in results])

            except Exception as e:
                if method == "vellanky" and target_fn in ["pendulum", "ackley"]:
                    continue  # skip missing results for vellanky on pendulum and ackley
                print(f"Error loading {method} {target_fn} l={lengthscale}: {e}")
                continue

            style = {"linestyle": linestyle, "color": color}
            plot_ys(ax_ys, ys, style=style)
            plot_times(ax_fit, t_fit, position=i)
            plot_times(ax_acq, t_acq, position=i)

    ax_ys.set(
        title="Target function",
        ylabel="Target",
        xlabel="Acquisitions",
        yscale="log",
    )
    ax_ys.grid(True)

    for ax, title in [
        (ax_fit, "Surrogate fit time"),
        (ax_acq, "Candidate search time"),
    ]:
        ax.set(title=title, ylabel="Time (seconds)", ylim=(0, None))
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45)
        ax.grid(True)

    # --- legend in its own axis to the right ---
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
    spacer = mlines.Line2D([], [], color="none", label="")
    all_handles = method_handles + [spacer] + profile_handles

    ax_leg.axis("off")
    ax_leg.legend(
        handles=all_handles,
        loc="center left",
        frameon=True,
        title="Method / Kernel",
    )

    fig.suptitle(f"{target_fn} ($\\rho={lengthscale}$)")
    fig.tight_layout()

    os.makedirs("plots", exist_ok=True)
    fig.savefig(f"plots/{target_fn}_lengthscale_{lengthscale}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # save_dir = f"results/wycoff_matern32/mnist/lengthscale_0.3/"
    # import jax
    # import jax.numpy as jnp
    # results = [
    #     np.load(os.path.join(save_dir, f), allow_pickle=True)
    #     for f in os.listdir(save_dir)
    #     if f.endswith(".pkl")
    # ]
    # fs = [r["observation_locations"] for r in results]
    # ys = [r["observation_values"] for r in results]
    # for f, y in zip(fs, ys):
    #     f = f[np.argmin(y)]
    #     a = lambda x: f((x[None] + 3.0) / 6.0) + jax.nn.celu(x)
    #     x = np.linspace(-10.0, 10.0, 100)
    #     plt.plot(x, jax.vmap(a)(x))
    #     break
    # plt.savefig("plots/mnist_surrogate.pdf", bbox_inches="tight")
    # assert False

    methods = ["wycoff", "wycoff_gp", "kundu", "vien", "shilton", "vellanky"]
    profiles = ["rbf", "matern52", "matern32"]

    for lenghtscale in [0.3, 0.1, 0.03]:
        for target_fn in ["sinc", "ackley", "pendulum", "mnist"]:
            print(f"Plotting results for {target_fn} (lengthscale {lenghtscale})")
            plot(target_fn, lenghtscale, methods, profiles)
            print(f"Done!\n")
