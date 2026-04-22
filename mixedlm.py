import pandas as pd
import statsmodels.formula.api as smf
import argparse


def fit_table(df, outcomes, reference="random", groups_col="group", verbose=False):
    rows = {}
    for outcome in outcomes:
        m = smf.mixedlm(
            # f"{outcome} ~ method - 1" # this disables intercept
            f"{outcome} ~ C(method, Treatment(reference='{reference}'))", # makes intercept = c(reference)
            df, 
            groups=df[groups_col],
        ).fit()
        if verbose:
            print(m.summary())
        rows[outcome] = m.params
    tbl = pd.DataFrame(rows)
    tbl.index = tbl.index.str.removeprefix("method[")
    tbl.index = tbl.index.str.removeprefix(f"C(method, Treatment(reference='{reference}'))[T.")
    tbl.index = tbl.index.str.removesuffix("]")
    tbl.index = tbl.index.str.replace("Intercept", reference)
    return tbl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-s", "--single_target", action="store_true")
    args = parser.parse_args()

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
    methods = ["random", "wycoff", "vien", "vellanky", "shilton", "kundu"]
    sub = sub[sub["method"].isin(methods)]
    print(fit_table(sub, outcomes, verbose=args.verbose).loc[methods].to_string())
    print("========================================")

    if not args.single_target:
        exit()

    for target_fn, subsub in sub.groupby("target_fn"):
        subsub = subsub[sub["method"].isin(methods)]
        print("========================================")
        print(f"{target_fn}")
        print()
        print(fit_table(subsub, outcomes, verbose=args.verbose).to_string())
    print("========================================")
