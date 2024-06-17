import matplotlib.pyplot as plt
import numpy as np

from treeffuser import Treeffuser

# Generate the data
seed = 0
rng = np.random.default_rng(seed=seed)
n = 5000
x = rng.uniform(0, 2 * np.pi, size=n)
comp = rng.integers(0, 2, size=n)
y = comp * np.sin(x - np.pi / 2) + (1 - comp) * np.cos(x) + rng.laplace(scale=x / 20, size=n)

# Fit the model
model = Treeffuser(seed=seed)
model.fit(x, y)

# Generate and plot samples
y_samples = model.sample(x, n_samples=1, seed=seed, verbose=True)

plt.scatter(x, y, s=1, label="raw data")
plt.scatter(x, y_samples[0, :], s=1, alpha=0.7, label="samples")
plt.ylim(-2.5, 2.5)

plt.xlabel("$x$")
plt.ylabel("$y$")

plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.125), ncol=2)
plt.tight_layout()
plt.savefig("README_example.png", dpi=120)
