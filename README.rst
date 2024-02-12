========
Overview
========

Diffusion but trees

* Free software: MIT license

Installation
============

::

    pip install treeffuser

You can also install the in-development version with::

    pip install git+ssh://git@https://github.com/blei-lab/tree-diffuser/blei-lab/treeffuser.git@main

Documentation
=============


https://treeffuser.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

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
