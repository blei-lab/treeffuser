====================
Treeffuser
====================

Treeffuser is an easy-to-use package for probabilistic prediction on tabular data with tree-based diffusion models.
Its goal is to estimate distributions of the form `p(y|x)` where `x` is a feature vector, `y` is a target vector
and the form of `p(y|x)` can be arbitrarily complex (e.g multimodal, heteroskedastic, non-gaussian, heavy-tailed, etc).

It is designed to adhere closely to the scikit-learn API and requires minimal user tuning.

Installation
============

You can install Treeffuser via pip from PyPI with the following command::

    pip install treeffuser

You can also install the in-development version with::

    pip install git+https://github.com/blei-lab/tree-diffuser.git@main


Usage Example
============

Here's a simple example demonstrating the usage of Treeffuser.

We generate an heteroscedastic response with a sinusoidal mean and fat tails.

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    from treeffuser import Treeffuser

    # Generate data
    rng = np.random.default_rng(seed=0)
    n = 5000
    x = rng.uniform(0, 2 * np.pi, size=n)
    y = np.sin(x) + rng.laplace(scale=x / 20, size=n)

We fit Treeffuser and generate samples. These can be used to compute any downstream estimates of interest.

.. code-block:: python

    # Fit the model
    model = Treeffuser()
    model.fit(x, y)

    # Generate samples and return downstream estimates
    x_new = np.linspace(x.min(), x.max(), 200)
    y_samples = model.sample(x_new, n_samples=10**2, verbose=True)
    y_preds = y_samples.mean(axis=0)
    y_q05, y_q95 = np.quantile(y_samples, q=[0.05, 0.95], axis=0)
    ... # other metrics

We then plot the original data along with the model's predictions.

.. code-block:: python

    sorted_idx = np.argsort(x_new)
    x_sorted, y_preds_sorted, y_q05_sorted, y_q95_sorted = [
        arr[sorted_idx] for arr in [x_new, y_preds, y_q05, y_q95]
    ]
    plt.plot(x_sorted, y_preds_sorted, color="black")
    plt.fill_between(x_sorted, y_q05_sorted, y_q95_sorted, color="gray", alpha=0.4)
    plt.scatter(x, y, s=1)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
.. image:: README_example.png
   :alt: Treeffuser on heteroscedastic data with sinuisodal response and fat tails
   :align: center

Please refer to the docstrings for more information on the available methods and parameters.
