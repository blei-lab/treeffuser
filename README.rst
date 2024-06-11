====================
Treeffuser
====================

Treeffuser is an easy-to-use package for probabilistic prediction on tabular data with tree-based diffusion models.
Its goal is to estimate distributions of the form `p(y|x)` where `x` is a feature vector, `y` is a target vector
and the form of `p(y|x)` can be arbitrarily complex (e.g multimodal, heteroskedastic, non-gaussian, heavy-tailed, etc).

It is designed to adhere closely to the scikit-learn API and requires minimal user tuning.

Usage Example
-------------

Here's how you can use Treeffuser in your project:

.. code-block:: python

    from treeffuser import LightGBMTreeffuser
    import numpy as np

    # (n_training, n_features), (n_training, n_targets)
    X, y = ...  # load your data
    # (n_test, n_features)
    X_test = ...  # load your test data

    # Estimate p(y|x) with a tree-based diffusion model
    model = LightGBMTreeffuser()
    model.fit(X, y)

    # Draw samples y ~ p(y|x) for each test point
    # (n_samples, n_test, n_targets)
    y_samples = model.sample(X_test, n_samples=1000)

    # Compute downstream metrics
    mean = np.mean(y_samples, axis=0)
    std = np.std(y_samples, axis=0)
    median = np.median(y_samples, axis=0)
    quantile = np.quantile(y_samples, q=0 axis=0)
    ... # other metrics

Please refer to the docstrings for more information on the available methods and parameters.

Installation
============

You can install Treeffuser via pip from PyPI with the following command::

    pip install treeffuser

You can also install the in-development version with::

    pip install git+https://github.com/blei-lab/tree-diffuser.git@main
