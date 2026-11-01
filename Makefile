PY ?= python3
K8S_VERSION ?= 1.34

.PHONY: help install all kvcalc capacity routesim validate lint test screenshots clean

help:
	@echo "make install      pip install -r requirements.txt"
	@echo "make all          run every tool once (kvcalc, capacity, routesim, validate)"
	@echo "make validate     offline schema-validate manifests (K8S_VERSION=$(K8S_VERSION))"
	@echo "make lint         yamllint the manifests"
	@echo "make test         pytest"
	@echo "make screenshots  regenerate docs/img/*.png from real tool output"

install:
	$(PY) -m pip install -r requirements.txt

all: kvcalc capacity routesim validate

kvcalc:
	$(PY) tools/kvcalc.py --ctx 8192

capacity:
	$(PY) tools/captool.py --qps 60 --e2e 6 --ramp-qps-per-min 20

routesim:
	$(PY) tools/routesim.py --lam 80

validate:
	$(PY) tools/validate.py $(K8S_VERSION)

lint:
	$(PY) -m yamllint -c .yamllint manifests

test:
	$(PY) -m pytest -q

screenshots:
	bash tools/make_screenshots.sh

clean:
	rm -rf out .pytest_cache tools/__pycache__ tests/__pycache__
