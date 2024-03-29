
from testbed.data.utils import get_data
from testbed.data.utils import list_data
from sklearn.model_selection import train_test_split

# %%
from testbed.models.ngboost_.ngboost_ import NGBoostGaussian, NGBoostMixtureGaussian
from testbed.models.treeffuser import Treeffuser
from testbed.metrics.accuracy import AccuracyMetric
from testbed.metrics.calibration import QuantileCalibrationErrorMetric


# %%
list_data()

# %%
data = get_data("protein", verbose=True)
# only take 1000 samples
data["x"] = data["x"][:1000]
data["y"] = data["y"][:1000]

X_train, X_test, y_train, y_test = train_test_split(
    data["x"], data["y"], test_size=0.2, random_state=42
)


model1 = NGBoostGaussian()
model2 = NGBoostMixtureGaussian()
model3 = Treeffuser()

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

# %%
for metric in [AccuracyMetric(), QuantileCalibrationErrorMetric()]:
    print(metric.compute(model1, X_test, y_test))
    print(metric.compute(model2, X_test, y_test))
    print(metric.compute(model3, X_test, y_test))
