Getting started
User Guide
API reference
Development
Release notes

Create a docs folder

# Set up Jekyll
Install Ruby and Bundler:
bash
Copy code
brew instlal ruby
echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
gem install jekyll bundler

Create a new Jekyll site in docs/jekyll:
bash
Copy code
jekyll new jekyll
cd jekyll

Add the "Just the Docs" theme to your Jekyll site by modifying the Gemfile:
ruby
Copy code
gem "just-the-docs"

Install the theme:
bash
Copy code
bundle install

Update your _config.yml file to use the "Just the Docs" theme:
yaml
Copy code
theme: just-the-docs

Optionally, configure the theme settings according to your preferences, like navigation, search, etc.

# Set up Sphinx
Create a docs/sphinx folder

Using Sphinx with recommonmark:
bash
Copy code
pip install sphinx recommonmark

sphinx-quickstart

In conf.py, configure Sphinx to use Markdown:
python
Copy code
extensions = [
    'recommonmark',
]

Use sphinx-markdown-builder to generate markdown files:
bash
Copy code
pip install sphinx-markdown-builder

sphinx-build -b markdown ./source ./docs

# Configuring Sphinx to Document Your Python Package

To ensure Sphinx can generate documentation from your Python package's docstrings, you must configure the conf.py file in the source directory:

Set the sys.path in conf.py to include the path to your Python package. This allows Sphinx to import your modules and access their docstrings.
Example configuration snippet in conf.py:
python
Copy code
import os
import sys
sys.path.insert(0, os.path.abspath('../../mypackage'))  # Adjust the path as necessary
Configure Sphinx to use autodoc, a handy extension for pulling documentation from docstrings:
python
Copy code
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon'  # If using Google or NumPy style docstrings
]
This setup keeps your Python package's source code and documentation logically separated yet integrated in a way that Sphinx can automatically generate documentation based on your code's docstrings. This organization is crucial for maintaining clarity and manageability in larger projects.

Generate Documentation
Create reStructuredText (.rst) files that Sphinx can process. The simplest way is to use sphinx-apidoc:
sh
Copy code
sphinx-apidoc -o sphinx/source ../../src  # Adjust the paths as needed
This will generate .rst files for your modules in the docs/source directory.

Build the HTML documentation with Sphinx:
sh
Copy code
sphinx-build -b html sphinx/source sphinx/build/html

# Alternatively run make html
make html
pip install furo


# Running Sphinx
Build and Serve Sphinx Documentation: Unlike Jekyll, Sphinx doesn’t have a built-in server for live previews, so you build the HTML with Sphinx and view it by opening the HTML files directly or setting up a simple HTTP server:
bash
Copy code
cd build
python -m http.server
This serves the Sphinx-generated documentation at a URL like http://localhost:8000.

# Furo theme
pip install furo

Add the following to conf.py
html_theme = "furo"  # "alabaster"

# Sphinx to generate automatically markdown files
https://www.sphinx-doc.org/en/master/usage/markdown.html

# Jekyll
Build and serve your Jekyll site locally to see your documentation:
sh
Copy code
bundle exec jekyll serve
