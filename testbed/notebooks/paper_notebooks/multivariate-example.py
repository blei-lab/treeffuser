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
from testbed.models.treeffuser import Treeffuser

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
                    y = np.array([j, height - i]) + np.random.randn(2) * 1
                    x = np.random.uniform(0.8, 1)

                    xs.append(x)
                    ys.append(y * x / height)

    x = np.array(xs)[:, None]
    Y = np.array(ys)

    return x, Y


x, Y = numpy_arr_to_dataset(img_array, 30)

print("shape length of x: ", x.shape)


# %%
x_large = x > 0.99
y_large = Y[x_large.flatten()]

plt.scatter(y_large[:, 0], y_large[:, 1], color="black", alpha=1, s=1)
# %%

model = Treeffuser()
model.fit(x, Y)

# %%
n_samples = 10000
x_test = np.ones((n_samples, 1)) * 0.8
samples = model.sample(x_test, 1)
print("shape of samples: ", samples.shape)

# %%
plt.scatter(samples[0, :, 0], samples[0, :, 1], color="black", alpha=1, s=1)
plt.xlim(0, 1)
plt.ylim(0, 1)

# %%
