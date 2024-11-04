Documentation
=============

This guide explains how to build Treeffuser's documentation.

The documentation uses Sphinx with the `Furo`_ theme.

Building the documentation
--------------------------

Navigate to the documentation folder, where this `README.rst` is located::

    cd treeffuser/docs/docs

Build the documentation::

    sphinx-build -b html source/ ./


Clearing build files
--------------------

To remove previously generated HTML files and clean up build artifacts, first ensure `clear_build.sh` is executable. If not, make it executable with::

    chmod +x clear_build.sh

Then run `clear_build.sh`:

    ./clear_build.sh

.. _Furo: https://github.com/pradyunsg/furo
