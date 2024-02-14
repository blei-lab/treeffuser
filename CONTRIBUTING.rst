============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

Bug reports
===========

When `reporting a bug <https://https://github.com/blei-lab/tree-diffuser/blei-lab/treeffuser/issues>`_ please include:

    * Your operating system name and version.
    * Any details about your local setup that might be helpful in troubleshooting.
    * Detailed steps to reproduce the bug.

Documentation improvements
==========================

treeffuser could always use more documentation, whether as part of the
official treeffuser docs, in docstrings, or even on the web in blog posts,
articles, and such.

Feature requests and feedback
=============================

The best way to send feedback is to file an issue at https://https://github.com/blei-lab/tree-diffuser/blei-lab/treeffuser/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that code contributions are welcome :)

Development
===========

To set up `treeffuser` for local development:

1. Fork `treeffuser <https://https://github.com/blei-lab/tree-diffuser/blei-lab/treeffuser>`_
   (look for the "Fork" button).
2. Clone your fork locally::

    git clone git@https://github.com/blei-lab/tree-diffuser:YOURGITHUBNAME/treeffuser.git

3. Create a branch for local development::

    git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

4. For installing the package run ::

   make

   This will create a virtual environment and install the package in it. To activate this environment run ::

   source .venv/bin/activate

5. When you're done making changes run all the checks and docs builder with one command::

    tox

6. Commit your changes and push your branch to GitHub::

    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

If you need some code review or feedback while you're developing the code just make the pull request.

For merging, you should:

1. Include passing tests (run ``tox``).
2. Update documentation when there's new API, functionality etc.
3. Add a note to ``CHANGELOG.rst`` about the changes.
4. Add yourself to ``AUTHORS.rst``.

Tips
----

To run a subset of tests::

    tox -e envname -- pytest -k test_myfeature

To run all the test environments in *parallel*::

    tox -p auto

Before committing it is important to run pre-commit (github will check that you
did). Running tox does this but if you want to dot it manually here is a short blurb
of what is and how it works.
`pre-commit` is tool that runs checks on your code before you commit it. It is great!
Here is the workflow on how to use it:
Assume there are files `file.txt` and `scripty.py`. Then the workflows is::

    git add file.txt
    git add scripty.py
    pre-commit
    ... [fix all of the things that can't be automatically fixed ] ...
    git add file.txt
    git add script.txt
    git commit -m "some message"
