# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
import pandas as pd
from data.utils import get_data
from data.utils import list_data
from metrics.accuracy import compare_accuracy
from metrics.calibration_scalar import compare_calibration
from ngboost import NGBRegressor
from sklearn.model_selection import train_test_split

from treeffuser import LightGBMTreeffuser

# %%
# ## Data

# %%
list_data()

# %%
# ## Models

# %%
N_SAMPLES = 64

for data_name in ["protein"]:
    data = get_data(data_name, verbose=True)
    # only take 1000 samples
    data["x"] = data["x"][:1000]
    data["y"] = data["y"][:1000]

    X_train, X_test, y_train, y_test = train_test_split(
        data["x"], data["y"], test_size=0.2, random_state=42
    )

    models = {}
    preds = {}
    samples = {}

    for sde_name in ["vesde"]:  # ["vpsde", "vesde", "sub-vpsde"]:
        model = LightGBMTreeffuser(
            verbose=1,
            n_repeats=100,
            n_estimators=10000,
            sde_name=sde_name,
            learning_rate=0.01,
            early_stopping_rounds=50,
            num_leaves=31,
        )
        temp = model.fit(X_train, y_train)
        y_samples = model.sample(
            X_test,
            n_samples=N_SAMPLES,
            n_parallel=8,
            n_steps=100,
            seed=0,
            verbose=1,
        )
        model_name = f"treeffuser_{sde_name}"
        models[model_name] = model
        preds[model_name] = y_samples.mean(axis=0)
        samples[model_name] = y_samples

    model_name = "ngb"
    model = NGBRegressor(early_stopping_rounds=50)
    model.fit(X_train, y_train)
    models[model_name] = model
    y_dim = data["y"].shape[1]
    preds[model_name] = model.predict(X_test).reshape((-1, y_dim))
    y_samples = model.pred_dist(X_test).sample(N_SAMPLES)
    samples[model_name] = y_samples

    metrics_accuracy = compare_accuracy(preds, y_test, print_table=False)
    metrics_calibration = compare_calibration(samples, y_test)

    # merge metrics
    metrics = pd.concat(
        [pd.DataFrame(metrics_accuracy), pd.DataFrame(metrics_calibration)], axis=0
    )

    break

metrics

# %%
from matplotlib import pyplot as plt

for label in ["treeffuser_vesde", "ngb"]:
    quantiles = []
    for i in range(y_test.shape[0]):
        y_true_i = y_test[i][0]
        quantile = np.mean(samples[label][:, i] <= y_true_i)
        quantiles.append(quantile)

    quantiles = np.sort(quantiles)
    x = np.linspace(0, 1, y_test.shape[0])
    print(np.abs(quantiles - x).mean())
    plt.plot(x, quantiles, label=label)
plt.plot(x, x, linestyle="--", color="black")
plt.legend()
plt.show()

# %%
metrics_accuracy = compare_accuracy(preds, y_test, print_table=False)
metrics_calibration = compare_calibration(samples, y_test)

# merge metrics
metrics = pd.concat(
    [pd.DataFrame(metrics_accuracy), pd.DataFrame(metrics_calibration)], axis=0
)

# %%
metrics

# %%

# %%

# %%
# # V data

# %%
n = 1000
X_train1 = np.random.rand(n, 1)
y_train1 = X_train1 + np.random.randn(n, 1) * 0.05 * (X_train1 + 1) ** 2

X_train2 = np.random.rand(n, 1)
y_train2 = -X_train2 + np.random.randn(n, 1) * 0.05 * (X_train2 + 1) ** 2

X_train = np.concatenate([X_train1, X_train2], axis=0)
y_train = np.concatenate([y_train1, y_train2], axis=0)

# %%
N_SAMPLES = 256

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

model = {}
preds = {}
samples = {}

for sde_name in ["vesde"]:  # ["vpsde", "vesde", "sub-vpsde"]:
    model = LightGBMTreeffuser(
        verbose=1,
        n_repeats=100,
        n_estimators=1000,
        sde_name=sde_name,
        learning_rate=0.1,
        early_stopping_rounds=50,
        num_leaves=31,
    )
    temp = model.fit(X_train, y_train)
    y_samples = model.sample(
        X_test,
        n_samples=N_SAMPLES,
        n_parallel=8,
        n_steps=30,
        seed=0,
        verbose=1,
    )
    model_name = f"treeffuser_{sde_name}"
    preds[model_name] = y_samples.mean(axis=0)
    samples[model_name] = y_samples

model_name = "ngb"
model = NGBRegressor(early_stopping_rounds=50)
model.fit(X_train, y_train)
y_dim = data["y"].shape[1]
preds[model_name] = model.predict(X_test).reshape((-1, y_dim))
y_samples = model.pred_dist(X_test).sample(N_SAMPLES)
samples[model_name] = y_samples

metrics_accuracy = compare_accuracy(preds, y_test, print_table=False)
metrics_calibration = compare_calibration(samples, y_test)

# merge metrics
metrics = pd.concat(
    [pd.DataFrame(metrics_accuracy), pd.DataFrame(metrics_calibration)], axis=0
)

metrics

# %%
from matplotlib import pyplot as plt

for label in ["treeffuser_vesde", "ngb"]:
    quantiles = []
    for i in range(y_test.shape[0]):
        y_true_i = y_test[i][0]
        quantile = np.mean(samples[label][:, i] <= y_true_i)
        quantiles.append(quantile)

    quantiles = np.sort(quantiles)
    x = np.linspace(0, 1, y_test.shape[0])
    print(np.abs(quantiles - x).mean())
    plt.plot(x, quantiles, label=label)
plt.plot(x, x, linestyle="--", color="black")

plt.legend()

# %%
import seaborn as sns

idx = 2
x = X_test[idx]

n_bins = 30
_n = 1000
y_bins = np.linspace(-2, 2, n_bins)
ground_truth = (
    x * (np.random.binomial(1, 0.5, _n) - 0.5) * 2 + np.random.randn(_n) * 0.05 * (x + 1) ** 2
)
sns.histplot(ground_truth, bins=y_bins, color="red", alpha=0.5, stat="density")

sns.histplot(samples["ngb"][:, idx].squeeze(), bins=y_bins, stat="density")
# sns.histplot(samples["treeffuser_vesde"][:,idx].squeeze(), bins=np.linspace(-1, 1, n_bins), stat='density')

# %%
# # Simulated calibration curve

# %%
ground_truth_x = np.random.uniform(-10, 10, 200)
ground_truth_y = np.random.normal(ground_truth_x, 1)
overconfident_model_y = np.random.normal(
    ground_truth_x, 0.2, size=(1000, ground_truth_x.shape[0])
)
underconfident_model_y = np.random.normal(
    ground_truth_x, 2, size=(1000, ground_truth_x.shape[0])
)

# %%
x = np.linspace(0, 1, ground_truth_y.shape[0])
for model_name, model_y in [
    ("overconfident", overconfident_model_y),
    ("underconfident", underconfident_model_y),
]:
    quantiles = []
    for i in range(ground_truth_y.shape[0]):
        ground_truth_y_i = ground_truth_y[i]
        quantile = np.mean(model_y[:, i] <= ground_truth_y_i)
        quantiles.append(quantile)

    quantiles = np.sort(quantiles)
    plt.plot(x, quantiles, label=model_name)

plt.plot(x, x, linestyle="--", color="black")
plt.legend()

# %%
