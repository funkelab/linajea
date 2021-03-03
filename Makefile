default:
	pip install -r requirements.txt
	pip install .
	-rm -rf dist build linajea.egg-info

.PHONY: install-dev
install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .
	-rm -rf dist build linajea.egg-info

.PHONY: tests
tests:
	PY_MAJOR_VERSION=py`python -c 'import sys; print(sys.version_info[0])'` pytest --cov-report term-missing -v --cov=linajea --cov-config=.coveragerc tests
	flake8 linajea
