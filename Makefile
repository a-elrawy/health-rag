.PHONY: setup run test eval examples clean

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup: $(VENV)/bin/activate
	$(PIP) install -r requirements.txt

$(VENV)/bin/activate:
	python3 -m venv $(VENV)

run: setup
	$(VENV)/bin/uvicorn app.main:app --reload

test: setup
	$(PY) -m pytest -q

eval: setup
	$(PY) -m eval.run_eval --write

examples: setup
	$(PY) -m scripts.run_examples

clean:
	rm -rf $(VENV) .pytest_cache app/__pycache__ tests/__pycache__ eval/__pycache__ scripts/__pycache__ eval/_eval_run_log.*
