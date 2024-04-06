#!/usr/bin/env python
from pathlib import Path

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with Path(__file__).parent.joinpath(*names).open(
        encoding=kwargs.get("encoding", "utf8")
    ) as fh:
        return fh.read()


setup(
    name="testbed",
    version="0.0.0",
    license="MIT",
    description="testbed for diffusion project",
    author="Achille Nazaret - Nicolas Beltran - Alessandro Grande",
    author_email="nb2838@columbia.edu",
    url="https://https://github.com/blei-lab/tree-diffuser/blei-lab/treeffuser",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[path.stem for path in Path("src").glob("*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        # uncomment if you test on these interpreters:
        # "Programming Language :: Python :: Implementation :: IronPython",
        # "Programming Language :: Python :: Implementation :: Jython",
        # "Programming Language :: Python :: Implementation :: Stackless",
        "Topic :: Utilities",
    ],
    project_urls={
        "Documentation": "https://treeffuser.readthedocs.io/",
        "Changelog": "https://treeffuser.readthedocs.io/en/latest/changelog.html",
        "Issue Tracker": "https://https://github.com/blei-lab/tree-diffuser/blei-lab/treeffuser/issues",
    },
    keywords=[
        # eg: "keyword1", "keyword2", "keyword3",
    ],
    python_requires=">=3.8",
    install_requires=[
        "matplotlib",
        "einops",
        "jaxtyping",
        "scikit-learn",
        "lightgbm",
        "tqdm",
        "ml-collections",
    ],
    extras_require={
        # eg:
        #   "rst": ["docutils>=0.11"],
        #   ":python_version=="2.6"": ["argparse"],
    },
)
