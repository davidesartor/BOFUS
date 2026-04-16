import os
import matplotlib.pyplot as plt
import numpy as np


def plot_ys(ax, ys: np.ndarray, label: str):
    runs, max_acquisitions = ys.shape
    y_best = np.minimum.accumulate(ys, axis=-1)
    x = np.arange(max_acquisitions)
    # ci on median
    qu = np.clip(0.5 + 1.96 * np.sqrt(0.25 / runs), 0, 1)
    ql = np.clip(0.5 - 1.96 * np.sqrt(0.25 / runs), 0, 1)
    ul = np.quantile(y_best, qu, axis=0)
    ll = np.quantile(y_best, ql, axis=0)
    ax.plot(x, np.median(y_best, axis=0), label=label)
    ax.fill_between(x, ll, ul, alpha=0.2, color=ax.lines[-1].get_color())


def plot_times(ax, times: np.ndarray, position: int):
    ax.boxplot(times, positions=[position], patch_artist=True)
    ax.set_ylabel(f"Time (seconds)")
    ax.grid(True)


def plot(target_fn: str, lengthscale: float, methods: list[str]):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    ax_ys, ax_fit, ax_acq = axes

    for i, method in enumerate(methods):
        save_dir = f"results/{method}/{target_fn}/lengthscale_{lengthscale}/"
        try:
            results = [
                np.load(os.path.join(save_dir, f), allow_pickle=True)
                for f in os.listdir(save_dir)
                if f.endswith(".pkl")
            ]

            ys = np.array([r["observation_values"] for r in results])
            t_fit = np.array([r["surrogate_fit_time"] for r in results])
            t_acq = np.array([r["acquisition_time"] for r in results])

        except Exception as e:
            print(f"Error loading {method} {target_fn} l={lengthscale}: {e}")
            continue

        plot_ys(ax_ys, ys, label=method)
        plot_times(ax_fit, t_fit, position=i)
        plot_times(ax_acq, t_acq, position=i)
        

    ax_ys.set(
        title="Target function",
        ylabel="Target",
        xlabel="Acquisitions",
        yscale="log",
    )
    ax_ys.legend()
    ax_ys.grid(True)


    for ax, title in [
        (ax_fit, "Surrogate fit time"),
        (ax_acq, "Candidate search time"),
    ]:
        ax.set(title=title, ylabel="Time (seconds)", ylim=(0, None))
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45)
        ax.grid(True)

    fig.suptitle(f"{target_fn} ($\\rho={lengthscale}$)")
    fig.tight_layout()

    os.makedirs("plots", exist_ok=True)
    fig.savefig(f"plots/{target_fn}_lengthscale_{lengthscale}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    methods = ["wycoff", "kundu", "vien", "shilton", "vellanky"]
    for lenghtscale in [0.1, 0.3, 0.03]:
        for target_fn in ["sinc", "ackley", "mnist", "pendulum"]:
            print(
                f"Plotting results for target function {target_fn} with lengthscale {lenghtscale}"
            )
            plot(target_fn, lenghtscale, methods)
            print(f"Done!\n")
