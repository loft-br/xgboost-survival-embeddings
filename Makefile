format:
	ruff format xgbse tests/ --check

lint:
	ruff check xgbse tests/

test:
	pytest --cov-report term-missing --cov=xgbse tests/

check: format lint test clean

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"
	pre-commit install

clean:
	rm -rf **/.ipynb_checkpoints **/.pytest_cache **/__pycache__ **/**/__pycache__ .ipynb_checkpoints .pytest_cache .coverage**
