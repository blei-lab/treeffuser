# Simple multivariate example
# %%
# load autoreload extension
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from numpy import ndarray
from jaxtyping import Float
from typing import Tuple
from treeffuser.treeffuser import LightGBMTreeffuser
from utils import set_plot_style


# %%
set_plot_style()

# %%

img = Image.open("misc/he.jpg")
rescale = 2
img = img.resize((img.size[0] // rescale, img.size[1] // rescale))

# %%
# Make it a numpy array
img_array = np.array(img)[:, :, 0]
height, width = img_array.shape
print("shape of the image array: ", img_array.shape)

# %%
# plot the image array
img_array = img_array / 255


# %%
def numpy_arr_to_dataset(
    arr: Float[ndarray, "height width"], n_repeat: int
) -> Tuple[Float[ndarray, "height width"], Float[ndarray, "height width"]]:
    """
    Converts a numpy array to a dataset in the following manner.

    We take x to be the scale of the image and y to be a two dimensional array
    indicating the coordinate of a black point.
    """
    xs = []
    ys = []

    height, width = arr.shape

    for _ in range(n_repeat):
        for i in range(height):
            for j in range(width):
                if arr[i, j] == 0:
                    x = np.random.uniform(0.5, 1)
                    y = (np.array([j, height - i]) / height + np.random.randn(1) * 0.001) * x
                    xs.append(x)
                    ys.append(y)

    x = np.array(xs)[:, None]
    Y = np.array(ys)

    return x, Y


x, Y = numpy_arr_to_dataset(img_array, 100)

print("shape length of x: ", x.shape)
print("shape length of Y: ", Y.shape)


# %%
x_large = x > 0.99
y_large = Y[x_large.flatten()]

plt.scatter(y_large[:, 0], y_large[:, 1], color="black", alpha=1, s=1)
# %%

model = LightGBMTreeffuser(
    sde_initialize_with_data=False,
    sde_name="vesde",
    n_estimators=20000,
    learning_rate=0.3,
    early_stopping_rounds=40,
    num_leaves=2000,
    max_depth=30,
    max_bin=1000,
    eval_percent=0.1,
    subsample_freq=0,
    subsample=1.0,
    verbose=2,
)
model.fit(x, Y)
"Model fitted successfully!"

# %%
n_samples = 10000
# x_test = np.random.uniform(0, 1, (n_samples, 1))
x_test = np.ones((n_samples, 1)) * 1
samples = model.sample(x_test, 1, n_steps=200, n_parallel=20)
print("shape of samples: ", samples.shape)

# %%
plt.scatter(samples[0, :, 0], samples[0, :, 1], color="black", alpha=1, s=1)
plt.xlim(0, 1)
plt.ylim(0, 1)

# %%

import numpy as np
import matplotlib.pyplot as plt


# Create the 3D Figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

x_coord = x_test.flatten()
y_coord = samples[0, :, 0].flatten()
z_coord = samples[0, :, 1].flatten()


# Scatter Plot
ax.scatter(
    x_coord, y_coord, z_coord, c=x_coord, marker="o", s=1, alpha=0.1
)  # x, y, z coordinates

# Axis Labels
ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
ax.set_zlabel("Z Label")
ax.view_init(elev=10, azim=0)

# Show the Plot
plt.show()


# %%
