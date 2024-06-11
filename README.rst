Treeffuser: Easy to use probabilistic prediction with tree-based diffusion models
==================================================================================

Treeffuser is an easy to use package for probabilistic prediction with tree-based diffusion models.
It is designed to adhere closely to the scikit-learn API and requires minimal user tuning.

Usage Example
=============

Here's how you can use Treeffuser in your project:

.. code-block:: python

    from treeffuser import Treeffuser
    import numpy as np

    # (n_training, n_features), (n_training, n_targets)
    X, y = ...  # load your data
    # (n_test, n_features)
    X_test = ...  # load your test data

    model = Treeffuser()
    model.fit(X, y)

    # (n_samples, n_test, n_targets)
    y_samples = model.sample(X_test, n_samples=1000)

    # Compute downstream metrics
    mean = np.mean(y_samples, axis=0)
    std = np.std(y_samples, axis=0)
    ... # other metrics

Installation
============

You can install Treeffuser via pip from PyPI with the following command::

    pip install treeffuser

You can also install the in-development version with::

    pip install git+https://github.com/blei-lab/tree-diffuser.git@main

Development
===========

To run all the tests, run::

    tox

However, this is usually excessive, so it is easier to use pytest with
your environment. When you push, tox will run automatically.

Note: To combine the coverage data from all the tox environments, run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
