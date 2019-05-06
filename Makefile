default:
	python setup.py install
	-rm -rf dist build linajea.egg-info

install-pip:
	pip install .
	-rm -rf dist build linajea.egg-info

.PHONY: install-full
install-full:
	pip install .[full]
	-rm -rf dist build linajea.egg-info

.PHONY: install-dev
install-dev:
	pip install -e .[full]
	-rm -rf dist build linajea.egg-info

.PHONY: tests
tests:
	PY_MAJOR_VERSION=py`python -c 'import sys; print(sys.version_info[0])'` pytest --cov-report term-missing -v --cov=linajea --cov-config=.coveragerc tests
	flake8 linajea
