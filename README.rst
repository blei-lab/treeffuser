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

You can also install the development version with::

    pip install git+https://github.com/blei-lab/tree-diffuser.git@main


Usage Example
============

Here's a simple example demonstrating how to use Treeffuser.

We generate an heteroscedastic response with two sinusoidal components and heavy tails.

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    from treeffuser import Treeffuser

    # Generate data
    seed=0
    rng = np.random.default_rng(seed=seed)
    n = 5000
    x = rng.uniform(0, 2 * np.pi, size=n)
    comp = rng.integers(0, 2, size=n)
    y = comp * np.sin(x - np.pi / 2) + (1 - comp) * np.cos(x) + rng.laplace(scale=x / 30, size=n)

We fit Treeffuser and generate samples. We then plot the samples against the raw data.

.. code-block:: python

    # Fit the model
    model = Treeffuser(seed=seed)
    model.fit(x, y)

    # Generate and plot samples
    y_samples = model.sample(x, n_samples=1, seed=seed, verbose=True)
    plt.scatter(x, y, s=1, label="raw data")
    plt.scatter(x, y_samples[0, :], s=1, alpha=0.7, label="samples")

.. image:: README_example.png
   :alt: Treeffuser on heteroscedastic data with sinuisodal response and heavy tails.
   :align: center

As Treeffuser recovers the target density, its samples can be used to compute any downstream estimates of interest.

.. code-block:: python

    # Compute downstream estimates
    y_samples = model.sample(x, n_samples=10**2, verbose=True)
    y_preds = y_samples.mean(axis=0)
    y_q05, y_q95 = np.quantile(y_samples, q=[0.05, 0.95], axis=0)
    ...

Please refer to the documentation for more information on the available methods and parameters.
