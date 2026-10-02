PY ?= python

.PHONY: install test lint data experiments analysis all

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	ruff check src tests

data:
	irteval prepare

experiments:
	for ds in d1 d2; do irteval rq2 $$ds && irteval rq1 $$ds && irteval budget $$ds; done

analysis:
	irteval analyze && irteval report

all: data experiments analysis
