import matplotlib.pyplot as plt
import numpy as np

from treeffuser import Treeffuser

# Generate the data
rng = np.random.default_rng(seed=0)
n = 5000
x = rng.uniform(0, 2 * np.pi, size=n)
y = np.sin(x) + rng.laplace(scale=x / 20, size=n)

# Fit the model
model = Treeffuser()
model.fit(x, y)

# Generate samples and return predictions
x_new = np.linspace(x.min(), x.max(), 200)
y_samples = model.sample(x_new, n_samples=10**2, verbose=True)
y_preds = y_samples.mean(axis=0)
y_q05, y_q95 = np.quantile(y_samples, q=[0.05, 0.95], axis=0)

# Plot original data and predictions
sorted_idx = np.argsort(x_new)
x_sorted, y_preds_sorted, y_q05_sorted, y_q95_sorted = (
    arr[sorted_idx] for arr in [x_new, y_preds, y_q05, y_q95]
)
plt.plot(x_sorted, y_preds_sorted, color="black")
plt.fill_between(x_sorted, y_q05_sorted, y_q95_sorted, color="gray", alpha=0.4)
plt.scatter(x, y, s=1)

plt.xlabel("x")
plt.ylabel("y")

plt.savefig("README_example.png", dpi=120)
