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

However, this is usually excessive so it is easier to use pytest with
your environment. When you push tox will run automatically. 

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



Before committing it is important to run pre-commit (github will check that you 
did). `pre-commit` will run automatically as a hook so that to commit things need
to adhere to the linter. To make sure this is the case you can use the following
work-stream. Assume there are files `file.txt` and `scripty.py`. Then the workflows is::

    git add file.txt
    git add scripty.py
    pre-commit
    ... [fix all of the things that can't be automatically fixed ] ...
    git add file.txt
    git add script.txt
    git commit -m "some message"

    

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
