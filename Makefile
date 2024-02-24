
# Run tests
test:
	python3 -m pytest -v

#creates a virtual environment and installs the required packages
make-dev:
	python3 -m venv .venv
	. .venv/bin/activate
	pip3 install -r requirements.txt
	pip3 install -e .
