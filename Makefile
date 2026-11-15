.DEFAULT_GOAL := all

PYTHON ?= .venv/bin/python

export SMOKE DRY_RUN RUNS

## all: the whole experiment (python main.py)
.PHONY: all
all:
	$(PYTHON) main.py

## smoke: the whole experiment, but shrunk to its smallest size (python main.py --smoke)
.PHONY: smoke
smoke:
	$(PYTHON) main.py --smoke

## dry-run: the whole experiment, but mock each environment's script call (python main.py --dry-run)
.PHONY: dry-run
dry-run:
	$(PYTHON) main.py --dry-run

## prices: run prices stage (python src/prices.py)
## estimate: run estimate stage (python src/estimate.py)
## measure: run measure stage (python src/measure.py)
## summarize: run summarize stage (python src/summarize.py)
## costs: run costs stage (python src/costs.py)
## charts: run charts stage (python src/charts.py)
.PHONY: prices estimate measure summarize costs charts
prices estimate summarize costs charts:
	$(PYTHON) src/$@.py
measure:
	$(PYTHON) src/measure.py

## clean: delete every result, after asking
.PHONY: clean
clean:
	@while :; do \
		printf 'Delete every file in out/? [y/N] '; read answer; \
		case "$$answer" in \
			[yY]|[yY][eE][sS]) rm -rf out/*; echo "deleted"; break ;; \
			""|[nN]|[nN][oO]) echo "kept"; break ;; \
		esac; \
	done

## help: show this menu
.PHONY: help
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /'
