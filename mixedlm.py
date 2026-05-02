import pandas as pd
import statsmodels.formula.api as smf
import argparse
import warnings


def fit_table(
    df,
    outcomes,
    method="mixedlm",
    reference="random",
    groups_col="group",
    verbose=False,
):
    methods_names = list(df.method.unique())
    if reference and reference not in methods_names:
        reference = None
    rows = {}
    for outcome in outcomes:
        formula = (
            f"{outcome} ~ C(method, Treatment(reference='{reference}'))*{groups_col}"
            if reference
            else f"{outcome} ~ method - 1"  # this disables the intercept
        )
        if method == "mixedlm":
            m = smf.mixedlm(formula, df, groups=df[groups_col]).fit()
        elif method == "anova":
            m = smf.ols(formula, df).fit()
        else:
            raise ValueError(f"Unknown method: {method}")
        if verbose:
            print(m.summary())

        conf = m.conf_int(alpha=0.05)
        rows[outcome] = m.params.map("{:.4g}".format) + conf.apply(
            lambda r: f" [{r[0]:.4g}, {r[1]:.4g}]", axis=1
        )
    tbl = pd.DataFrame(rows)
    if reference:
        tbl.index = tbl.index.str.replace("Intercept", reference)
        tbl.index = tbl.index.str.removeprefix(
            f"C(method, Treatment(reference='{reference}'))[T."
        )
    else:
        tbl.index = tbl.index.str.removeprefix("method[")
    tbl.index = tbl.index.str.removesuffix("]")
    tbl = tbl.loc[methods_names]
    return tbl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-s", "--silent", action="store_true")
    parser.add_argument("-t", "--single_target", action="store_true")
    parser.add_argument(
        "-m", "--method", type=str, choices=["mixedlm", "anova"], default="mixedlm"
    )
    args = parser.parse_args()
    if args.silent:
        warnings.filterwarnings("ignore")

    df = pd.read_csv("results.csv")
    outcomes = ["best_y", "avg_regret"]
    df["group"] = (
        df["target_fn"] + "_" + df["profile"] + "_" + df["lengthscale"].astype(str)
    )

    print("========================================")
    # use no_natural_grad version of the methods
    sub = df.copy()
    sub = sub[~sub["method"].isin(["wycoff", "vien"])]
    sub = sub.replace("wycoff_no_natural_grad", "wycoff")
    sub = sub.replace("vien_no_natural_grad", "vien")

    # only show the methods we care about
    methods = ["wycoff", "vien", "vellanky", "shilton", "kundu"]
    sub = sub[sub["method"].isin(methods)]
    print(fit_table(sub, outcomes, method=args.method, verbose=args.verbose))
    print("========================================")

    if args.single_target:
        for target_fn, subsub in sub.groupby("target_fn"):
            subsub = subsub[subsub["method"].isin(methods)]
            print(f"{target_fn}")
            print(fit_table(subsub, outcomes, method=args.method, verbose=args.verbose))
            print()

    print("========================================")
    # only show the methods we care about
    methods = [
        "wycoff",
        "wycoff_no_natural_grad",
        "vien",
        "vien_no_natural_grad",
    ]
    sub = df.copy()
    sub = sub[sub["method"].isin(methods)]
    print(fit_table(sub, outcomes, method=args.method, verbose=args.verbose))
    print("========================================")

    if args.single_target:
        for target_fn, subsub in sub.groupby("target_fn"):
            subsub = subsub[subsub["method"].isin(methods)]
            print(f"{target_fn}")
            print(fit_table(subsub, outcomes, method=args.method, verbose=args.verbose))
            print()
