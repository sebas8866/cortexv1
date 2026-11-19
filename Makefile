.PHONY: install test lint demo serve worker dashboard

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check . && ruff format --check . && mypy src

demo:
	python examples/ads_daily_review.py

serve:
	cortex serve

worker:
	cortex worker

dashboard:
	streamlit run dashboard/app.py
