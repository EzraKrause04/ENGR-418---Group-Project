# Latte Coach -- every result in the report is produced by a target here.
#
#   make setup                      create .venv (Python 3.11) and install pinned deps
#   make schedule SESSION=S05 POURER=A BLOCKS=2
#                                   plan a randomized session, append it to data/cup_log.csv
#   make ingest                     data/raw_full -> data/raw (EXIF stripped, 1024 px)
#   make label                      open the blind labelling tool (Gradio)
#   make qa                         contact sheets of every cup circle and crop (ARGS=--dev: dev set)
#   make circles                    check each circle by eye: accept it or click a fix
#   make features                   segment milk vs crema, handcrafted features -> features/features.csv
#   make gonogo                     go/no-go check (section 16): box plots, LOSO accuracy, score spread
#   make agree                      inter-rater agreement tables
#   make test                       run the test suite (must pass before modelling)
#   make all                        regenerate every table and figure from data/raw
#
# Written for GNU Make 3.81 (the macOS default), so no 3.82+ features.
# Recipes call $(PYTHON) directly instead of `source .venv/bin/activate`:
# activate breaks when the checkout path contains a ':' (it splits $PATH).

SHELL := /bin/bash
.DEFAULT_GOAL := help

VENV := .venv
ifeq ($(OS),Windows_NT)
  PYTHON := $(VENV)/Scripts/python.exe
else
  PYTHON := $(VENV)/bin/python
endif
PY311  ?= python3.11
UV     := $(shell command -v uv 2>/dev/null)
CONFIG ?= configs/default.yaml

# One BLAS thread for result targets: multi-threaded reductions sum in
# nondeterministic order, so reruns would differ in the last bits.
export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1
# Gradio phones home with usage analytics unless told not to.
export GRADIO_ANALYTICS_ENABLED := False

SESSION ?=
POURER  ?=
BLOCKS  ?= 2
ARGS    ?=

.PHONY: help setup venv-check schedule ingest label circles qa features embed \
        agree gonogo train final figures demo test env all clean

help:
	@sed -n '2,17p' Makefile | sed 's/^# \{0,1\}//'

# --- environment ------------------------------------------------------------

setup:
ifdef UV
	test -x $(PYTHON) || uv venv --seed --python 3.11 $(VENV)
	uv pip install --python $(PYTHON) -r requirements.txt
	uv pip install --python $(PYTHON) --no-deps -e .
else
	test -x $(PYTHON) || $(PY311) -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --no-deps -e .
endif
	$(PYTHON) -c "import latte, numpy, pandas, sklearn, cv2, skimage, torch, timm, gradio; print('setup OK: latte', latte.__version__)"

venv-check:
	@test -x $(PYTHON) || { echo "No virtualenv at $(VENV). Run: make setup"; exit 1; }

env: venv-check
	@mkdir -p results
	$(PYTHON) -m pip freeze --exclude-editable > results/env.txt
	@echo "wrote results/env.txt"

# --- data collection and labelling -----------------------------------------

schedule: venv-check
	@test -n "$(SESSION)" -a -n "$(POURER)" || { echo "usage: make schedule SESSION=S05 POURER=A [BLOCKS=2]"; exit 1; }
	$(PYTHON) scripts/make_schedule.py --config $(CONFIG) --session $(SESSION) --pourer $(POURER) --blocks $(BLOCKS) $(ARGS)

ingest: venv-check
	$(PYTHON) scripts/ingest.py --config $(CONFIG)

label: venv-check
	$(PYTHON) app/label.py --config $(CONFIG)

circles: venv-check
	$(PYTHON) app/annotate_circle.py --config $(CONFIG) $(ARGS)

agree: venv-check
	$(PYTHON) scripts/agreement.py --config $(CONFIG)

# --- image pipeline and modelling ------------------------------------------

qa: venv-check
	$(PYTHON) scripts/qa_sheets.py --config $(CONFIG) $(ARGS)

features: venv-check
	$(PYTHON) scripts/build_features.py --config $(CONFIG) $(ARGS)

embed: venv-check
	$(PYTHON) -m latte.embed --config $(CONFIG)

gonogo: venv-check
	$(PYTHON) scripts/go_no_go.py --config $(CONFIG)

train: venv-check
	$(PYTHON) scripts/train_eval.py --config $(CONFIG)

final: venv-check
	$(PYTHON) scripts/fit_final.py --config $(CONFIG)

figures: venv-check
	$(PYTHON) scripts/make_figures.py --config $(CONFIG)

demo: venv-check
	$(PYTHON) app/demo.py --config $(CONFIG)

test: venv-check
	$(PYTHON) -m pytest

# Full regeneration from data/raw (data/raw_full is not in the repo, so no ingest).
all: test env qa features embed agree train final figures

clean:
	rm -rf .pytest_cache src/*.egg-info
	find . -name __pycache__ -type d -prune -not -path './.venv/*' -exec rm -rf {} +
