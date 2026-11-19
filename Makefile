VENV := .venv
PYTHON := $(VENV)/bin/python

.PHONY: install run index test test-all

install: $(VENV)/.installed

$(VENV)/.installed: requirements.txt requirements-eval.txt requirements-dev.txt
	python3.11 -m venv $(VENV)
	$(VENV)/bin/pip install -q -r requirements-dev.txt
	touch $@

run: install
	$(VENV)/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000

index: install
	$(PYTHON) -m scripts.create_index

test: install
	$(VENV)/bin/pytest -v

test-all: install
	$(VENV)/bin/pytest -v tests/
