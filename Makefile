black:
	black xgbse setup.py --check

flake:
	flake8 xgbse setup.py

test:
	pytest --cov-report term-missing --cov=xgbse tests/

check: black flake test clean

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"
	pre-commit install

clean:
	rm -rf **/.ipynb_checkpoints **/.pytest_cache **/__pycache__ **/**/__pycache__ .ipynb_checkpoints .pytest_cache .coverage**
