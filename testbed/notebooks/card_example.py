# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
from testbed.models.card import Card
import matplotlib.pyplot as plt

# %%

# %%

n_features = 1
n_samples = 1000
X = np.random.rand(n_samples, n_features)
beta = np.random.rand(n_features, 1) * 10
std = 0.1
epsilon = np.random.randn(n_samples, 1) * std

y = X @ beta + epsilon

# plt.scatter(X, y)

# %%
# ## Models

model = Card(max_epochs=500, enable_progress_bar=True, n_steps=100)


# %%
model.fit(X, y)
# %%
y = model.predict(X)
plt.scatter(X, y)

# %%

samples = model.sample(X, 1).flatten()
print(samples.shape)
plt.scatter(X, samples)
