# %%

# %%
from data.utils import get_data
from data.utils import list_data
from metrics import compare_accuracy
from ngboost import NGBRegressor
from sklearn.model_selection import train_test_split

from treeffuser import LightGBMTreeffuser

# %%
# ## Data

# %%
list_data()

# %%
data = get_data("naval", verbose=True)
print(data.keys())
print(f"Categorical variables: {data['categorical']}")

# %%
X_train, X_test, y_train, y_test = train_test_split(
    data["x"], data["y"], test_size=0.2, random_state=42
)

# %%
# ## Models

# %%
model = {}
preds = {}

# %%
# First, we fit treeffuser.

# %%
model["treeffuser"] = LightGBMTreeffuser(
    verbose=1,
    n_repeats=100,
    n_estimators=10000,
    sde_name="vesde",
    learning_rate=0.1,
    early_stopping_rounds=50,
)
temp = model["treeffuser"].fit(
    X_train, y_train
)  # "temp=" is a temp fix for Issue #26, see github.com/blei-lab/tree-diffuser/issues/26

# %%
# We then sample from the fitted model.

# %%
y_samples = model["treeffuser"].sample(X_test, n_samples=1, n_parallel=100, n_steps=30, seed=0)
preds["treeffuser"] = y_samples.mean(axis=1)

# %%
# Next, we run NGBoost with Gaussian likelihood.

# %%
model["ngb"] = NGBRegressor(early_stopping_rounds=50)
model["ngb"].fit(X_train, y_train)
y_dim = data["y"].shape[1]
preds["ngb"] = model["ngb"].predict(X_test).reshape((-1, y_dim))

# %%
# ## Metrics

# %%
metrics = compare_accuracy(preds, y_test, print_table=True)
