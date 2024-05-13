"""
Importing Treeffuser causes segmentation fault of CARD. It is unclear why this
is happening as the two models are not related.

Therefore both models can't be run in the same script.
"""

import lightgbm as lgb  # noqa F401
from collections import defaultdict

import pandas as pd

# %%
# uncomment for interactivity
# %load_ext autoreload
# %autoreload 2

from testbed.data.utils import get_data
from testbed.data.utils import list_data
from sklearn.model_selection import train_test_split

# %%
from testbed.models.ngboost import NGBoostGaussian

from testbed.models.treeffuser import Treeffuser  # (causes segmentation fault ????)
from testbed.metrics.accuracy import AccuracyMetric
from testbed.metrics import CRPS


# %%
list_data()

# %%
# data = get_data("protein", verbose=True)
# data["x"] = data["x"][:1000]
# data["y"] = data["y"][:1000]

data = get_data("bike", verbose=True)
data["x"] = data["x"]
data["y"] = data["y"]

X_train, X_test, y_train, y_test = train_test_split(
    data["x"], data["y"], test_size=0.2, random_state=0
)


# %%

model1 = NGBoostGaussian()
model2 = Treeffuser()  # (causes segmentation fault of CARD ????)
# model3 = Card(max_epochs=5000, enable_progress_bar=True, n_steps=100)

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

# %%

results = defaultdict(dict)

for metric in [CRPS(100), AccuracyMetric()]:
    for model in [model1, model2]:
        model_name = model.__class__.__name__
        metric_model = metric.compute(model, X_test, y_test)
        results[model_name].update(metric_model)

results = pd.DataFrame(results)
print(results)
# %%
