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

.PHONY: test
test:
	python -m tests -v
