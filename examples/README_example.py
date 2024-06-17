import matplotlib.pyplot as plt
import numpy as np

from treeffuser import Treeffuser

# Generate the data
seed = 0
n = 5000
rng = np.random.default_rng(seed=seed)
x = rng.uniform(0, 2 * np.pi, size=n)
comp = rng.integers(0, 2, size=n)
y = comp * np.sin(x - np.pi / 2) + (1 - comp) * np.cos(x) + rng.laplace(scale=x / 30, size=n)

# Fit the model
model = Treeffuser(sde_initialize_from_data=True, seed=seed)
model.fit(x, y)

# Generate and plot samples
y_samples = model.sample(x, n_samples=1, seed=seed, verbose=True)

plt.scatter(x, y, s=1, label="observed data")
plt.scatter(x, y_samples[0, :], s=1, alpha=0.7, label="Treeffuser samples")

plt.xlabel("$x$")
plt.ylabel("$y$")

legend = plt.legend(loc="upper center", scatterpoints=1, bbox_to_anchor=(0.5, -0.125), ncol=2)
for legend_handle in legend.legend_handles:
    legend_handle.set_sizes([32])  # change marker size for legend

plt.tight_layout()

plt.savefig("README_example.png", dpi=120)
