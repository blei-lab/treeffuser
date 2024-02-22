#!/bin/bash
# This script uses pip to freeze the current state of the requirements
# into a requirements.txt file. We use this file to pin the versions of
# the dependencies we use in the project.

# The script should be run from the root of the project.
# The requirements.txt file will be created in ci/ directory

pip freeze > ci/requirements.txt
# Get rid of the git+ prefix in the requirements.txt file
sed -i '' 's/git+//g' ci/requirements.txt
# Get rid of the version number in the requirements.txt file
sed -i '' 's/==.*//g' ci/requirements.txt
# Get rid of anything with -e in the requirements.txt file
sed -i '' '/-e/d' ci/requirements.txt

# Append additional required packages with their version constraints
echo "virtualenv>=16.6.0" >> ci/requirements.txt
echo "pip>=19.1.1" >> ci/requirements.txt
echo "setuptools>=18.0.1" >> ci/requirements.txt
echo "six>=1.14.0" >> ci/requirements.txt
echo "tox" >> ci/requirements.txt
echo "twine" >> ci/requirements.txt

# Echo the contents of the requirements.txt file
echo "--------------------------------------------"
echo "Contents of ci/requirements.txt:"
echo "--------------------------------------------"
cat ci/requirements.txt

# Print the success message
echo "--------------------------------------------"
echo "Requirements have been frozen successfully!"
echo "--------------------------------------------"
