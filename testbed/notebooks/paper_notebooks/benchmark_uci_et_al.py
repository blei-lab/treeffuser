# %%
# Load the results from wandb csv files, and format them into a LaTeX table

# %%

import pandas as pd
from pathlib import Path
import numpy as np

# %%
path = Path("uci_results")
files = list(path.glob("*.csv"))

dfs = []
for file in files:
    df = pd.read_csv(file)
    dfs.append(df)

df = pd.concat(dfs)

# add _default to the model_name in rows were bayes_opt_score is NaN
# this is when we did not run bayes_opt
df.loc[df["bayes_opt_score"].isna(), "model"] += "_default"

# remove boston dataset
df = df[df["dataset"] != "boston"]


# %%


dataset_sizes = {
    "bike": 17379,
    # "boston": 506,
    "energy": 768,
    "naval": 11934,
    "news": 39644,
    "power": 9568,
    "superconductor": 21263,
    "wine": 6497,
    "yacht": 308,
    "movies": 7415,
    "kin8nm": 8192,
}

dataset_dim_x = {
    "bike": 12,
    # "boston": 13,
    "energy": 8,
    "naval": 17,
    "news": 58,
    "power": 4,
    "superconductor": 81,
    "wine": 12,
    "yacht": 6,
    "movies": 9,
    "kin8nm": 8,
}

dataset_dim_y = {
    "bike": 1,
    # "boston": 1,
    "energy": 2,
    "naval": 1,
    "news": 1,
    "power": 1,
    "superconductor": 1,
    "wine": 1,
    "yacht": 1,
    "movies": 1,
    "kin8nm": 1,
}

# %%

# Each dataset has a CPRS on a different scale, we will factor power of 10 out
# to make a more readable table

# change metric for rmse, mae, mace ...
metric = "crps_100"

tmp_results = pd.pivot_table(
    df,
    index=["dataset"],
    columns=["model"],
    values=[metric],
    aggfunc="mean",
)[metric]["treeffuser"]

log10_per_dataset = tmp_results.apply(lambda x: np.floor(np.log10(x)))

for dataset in log10_per_dataset.index:
    df.loc[df["dataset"] == dataset, metric] /= 10 ** log10_per_dataset[dataset]

# %%
tmp_results

# %%


def custom_aggfunc(x):
    mean, std = x.mean(), x.std()
    return [mean, std, False]
    # return rf"${mean:.02f} \scriptsize $\pm$ {std:.02f}$"


results = pd.pivot_table(
    df,
    index=["dataset"],
    columns=["model"],
    values=[metric],
    aggfunc=custom_aggfunc,
)[metric]

# find the best 2 method for each dataset, set the boolean (cell[2]) to True if it is one of the best 2
for dataset in results.index:
    best_2 = (
        results.loc[dataset]
        .apply(lambda x: x[0] if isinstance(x, list) else np.inf)
        .nsmallest(2)
    )
    print(best_2)
    results.loc[dataset] = results.loc[dataset].apply(
        lambda x: [x[0], x[1], x[0] in best_2.values] if isinstance(x, list) else x
    )


# format each cell
def format_cell(x):
    if isinstance(x, list):
        # if x[0] > 100000000000:
        #     print(x[0])
        #     return r"$\times$"
        if x[2]:
            return rf"$\mathbf{{{x[0]:.02f} {{\scriptstyle \pm {x[1]:.02f}}}}}$"
        return rf"${x[0]:.02f} {{\scriptstyle \pm {x[1]:.02f}}}$"
    else:
        return "--"


results = results.map(format_cell)

# energy has multivariate y so some methods cannot run
results.loc["energy"] = results.loc["energy"].apply(
    lambda x: r"\texttt{NA}" if x == "--" else x
)
# ngboost fails sometimes
results.loc[:, "ngboost"] = results.loc[:, "ngboost"].apply(
    lambda x: r"$\times$" if x == "--" else x
)
results.loc["news", "ngboost"] = r"$\times$"

results[" "] = log10_per_dataset.apply(lambda x: rf"$\scriptstyle\times 10^{{{int(x)}}}$")
results["$N$"] = results.index.map(dataset_sizes)
results["$d_x$"] = results.index.map(dataset_dim_x)
results["$d_y$"] = results.index.map(dataset_dim_y)

results["$N, d_x, d_y$"] = results[["$N$", "$d_x$", "$d_y$"]].apply(
    lambda x: "{:.0f},\\hfill {:.0f},\\hfill {:.0f}".format(*x), axis=1
)


methods = {
    "deep_ensemble": "Deep Ens.",
    "ngboost": "NGBoost",
    "ibug": "iBUG",
    "quantile_regression_tree": "QReg",
    "drf": "DRF",
    "treeffuser": "Treeffuser",
    "treeffuser_default": "Treeffuser*",
}
results = results[["$N, d_x, d_y$"] + list(methods.keys()) + [" "]]
results = results.rename(columns=methods)
results = results.rename(index={"superconductor": "superc."})

latex_table = results.to_latex(
    escape=False, column_format="l" + "c" * len(results.columns), index=True
)
latex_table = "\\resizebox{\\columnwidth}{!}{% \n " + latex_table + "}"


print(latex_table)
# %%
df["Runtime"].sum() // 3600, df["Runtime"].sum() // 60 % 60, df["Runtime"].sum() % 60

# %%
# Now synthetic datasets


# %%
path = Path("uci_results/syntheticv2")
files = list(path.glob("*.csv"))

dfs = []
for file in files:
    df = pd.read_csv(file)
    dfs.append(df)

df = pd.concat(dfs)

df = df[df["dataset"] != "student_t"]

# add _default to the model_name in rows were bayes_opt_score is NaN
df.loc[df["bayes_opt_score"].isna(), "model"] += "_default"

metric = "rmse"
# metric = "rmse"
# metric = "rmse"

tmp_results = pd.pivot_table(
    df,
    index=["dataset", "dataset_size", "dim_input"],
    columns=["model"],
    values=[metric],
    aggfunc="mean",
)[metric]["treeffuser"]

log10_per_dataset = tmp_results.apply(lambda x: np.floor(np.log10(x)))

for dataset in log10_per_dataset.index:
    df.loc[(df[["dataset", "dataset_size", "dim_input"]] == dataset).all(axis=1), metric] /= (
        10 ** log10_per_dataset[dataset]
    )


# %%


def custom_aggfunc(x):
    mean, std = x.mean(), x.std()
    return [mean, std, False]
    # return rf"${mean:.02f} \scriptsize $\pm$ {std:.02f}$"


results = pd.pivot_table(
    df,
    index=["dataset", "dataset_size", "dim_input"],
    columns=["model"],
    values=[metric],
    aggfunc=custom_aggfunc,
)[metric]

# find the best 2 method for each dataset, set the boolean (cell[2]) to True if it is one of the best 2
for dataset in results.index:
    best_2 = (
        results.loc[dataset]
        .apply(lambda x: x[0] if isinstance(x, list) else np.inf)
        .nsmallest(2)
    )
    print(best_2)
    results.loc[dataset] = results.loc[dataset].apply(
        lambda x: [x[0], x[1], x[0] in best_2.values] if isinstance(x, list) else x
    )


# format each cell
def format_cell(x):
    if isinstance(x, list):
        if x[0] > 200:
            # print(x[0])
            return r"\faWarning"
        if x[2]:
            return rf"$\mathbf{{{x[0]:.02f} {{\scriptstyle \pm {x[1]:.02f}}}}}$"
        return rf"${x[0]:.02f} {{\scriptstyle \pm {x[1]:.02f}}}$"
    else:
        return "--"


results = results.map(format_cell)

results[" "] = log10_per_dataset.apply(lambda x: rf"$\scriptstyle\times 10^{{{int(x)}}}$")
results = results.reset_index()


methods = {
    "deep_ensemble": "Deep Ens.",
    "ngboost": "NGBoost",
    "ngboost_poisson": "NGBoost Poisson",
    "ibug": "iBUG",
    "quantile_regression_tree": "QReg",
    "drf": "DRF",
    "treeffuser": "Treeffuser",
    "treeffuser_default": "Treeffuser*",
    "dataset_size": "$N$",
    "dim_input": "$d_x$",
}
# results = results[["$N, d_x, d_y$"] + list(methods.keys()) + [" "]]
results = results.rename(columns=methods)
# results = results.rename(index={"superconductor": "superc."})
results = results.drop("dataset", axis=1)

latex_table = results.to_latex(
    escape=False, column_format="l" + "c" * len(results.columns), index=False
)
latex_table = "\\resizebox{\\columnwidth}{!}{% \n " + latex_table + "}"


print(latex_table)

# %%
df["Runtime"].sum() // 3600, df["Runtime"].sum() // 60 % 60, df["Runtime"].sum() % 60, df[
    "Runtime"
].shape

# %%

# Now M5 dataset


# %%
path = Path("uci_results/m5")
files = list(path.glob("*.csv"))

dfs = []
for file in files:
    df = pd.read_csv(file)
    dfs.append(df)

df = pd.concat(dfs)

df = df[df["dataset"] != "student_t"]

# add _default to the model_name in rows were bayes_opt_score is NaN
df.loc[df["bayes_opt_score"].isna(), "model"] += "_default"

metric = "crps_100"
# metric = "rmse"
# metric = "mace"

tmp_results = pd.pivot_table(
    df,
    index=["dataset"],
    columns=["model"],
    values=[metric],
    aggfunc="mean",
)[metric]["treeffuser"]

log10_per_dataset = tmp_results.apply(lambda x: np.floor(np.log10(x)))

for dataset in log10_per_dataset.index:
    df.loc[df["dataset"] == dataset, metric] /= 10 ** log10_per_dataset[dataset]

# %%


def custom_aggfunc(x):
    mean, std = x.mean(), x.std()
    return [mean, std, False]
    # return rf"${mean:.02f} \scriptsize $\pm$ {std:.02f}$"


results = []
for metric in [
    "crps_100",
    "rmse",
    "mae",
    "mace",
]:
    results.append(
        pd.pivot_table(
            df,
            index=["dataset"],
            columns=["model"],
            values=[metric],
            aggfunc=custom_aggfunc,
        )[metric]
    )
    # add name of the metric in a column
    results[-1]["metric"] = metric

results = pd.concat(results, axis=0)
results = results.reset_index(drop=True).set_index("metric")

# find the best 2 method for each dataset, set the boolean (cell[2]) to True if it is one of the best 2
for dataset in results.index:
    best_2 = (
        results.loc[dataset]
        .apply(lambda x: x[0] if isinstance(x, list) else np.inf)
        .nsmallest(2)
    )
    print(best_2)
    results.loc[dataset] = results.loc[dataset].apply(
        lambda x: [x[0], x[1], x[0] in best_2.values] if isinstance(x, list) else x
    )


# format each cell
def format_cell(x):
    if isinstance(x, list):
        if x[0] > 200:
            # print(x[0])
            return r"\faWarning"
        if x[2]:
            return rf"$\mathbf{{{x[0]:.02f} }}$"
        return rf"${x[0]:.02f}$"
    else:
        return "--"


results = results.map(format_cell)

results = results.reset_index()

# results["$N, d_x, d_y$"] = results[["$N$", "$d_x$", "$d_y$"]].apply(
#     lambda x: "{:.0f},\\hfill {:.0f},\\hfill {:.0f}".format(*x), axis=1
# )


methods = {
    # "card": "CARD",
    "deep_ensemble": "Deep Ens.",
    "ngboost": "NGBoost",
    "ngboost_poisson": "NGBoost Poisson",
    "ibug": "iBUG",
    "quantile_regression_tree": "QReg",
    "drf": "DRF",
    "treeffuser": "Treeffuser",
    "treeffuser_default": "Treeffuser*",
    "dataset_size": "$N$",
    "dim_input": "$d_x$",
}
# results = results[["$N, d_x, d_y$"] + list(methods.keys()) + [" "]]
results = results.rename(columns=methods)
# results = results.rename(index={"superconductor": "superc."})
results = results.set_index(["metric"])


latex_table = results.to_latex(
    escape=False, column_format="l" + "c" * len(results.columns), index=True
)
latex_table = "\\resizebox{\\columnwidth}{!}{% \n " + latex_table + "}"


print(latex_table)
