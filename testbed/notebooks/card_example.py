# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
from testbed.models import Card
import matplotlib.pyplot as plt

# %%
# ## Data

# %%

X = np.random.rand(1000, 1)
beta = np.random.rand(1, 1) * 10
std = 0.1
epsilon = np.random.randn(1000, 1) * std

y = X @ beta + epsilon

plt.scatter(X, y)

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
# %%
