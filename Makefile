# Optional thin wrapper. The real entry point is `python -m driftguard.pipeline`.
PY ?= python

.PHONY: data train stream eval all app test lint

data:
	$(PY) -m driftguard.pipeline --data

train:
	$(PY) -m driftguard.pipeline --train

stream:
	$(PY) -m driftguard.pipeline --stream

eval:
	$(PY) -m driftguard.pipeline --eval

all:
	$(PY) -m driftguard.pipeline --all

app:
	$(PY) -m streamlit run app/app.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
