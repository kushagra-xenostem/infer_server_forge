.PHONY: install test lint format types bench demo report

install:
	uv sync --all-extras

test:
	uv run pytest -q

lint:
	uv run ruff check src tests packages tools/demo.py benchmarks/scripts/load_report.py benchmarks/scripts/failure_injection.py
	uv run ruff format --check src tests packages tools/demo.py benchmarks/scripts/load_report.py benchmarks/scripts/failure_injection.py

format:
	uv run ruff format src tests packages tools/demo.py benchmarks/scripts/load_report.py benchmarks/scripts/failure_injection.py
	uv run ruff check --fix src tests packages tools/demo.py benchmarks/scripts/load_report.py benchmarks/scripts/failure_injection.py

types:
	uv run mypy

bench:  ## smoke benchmark: a short simulator run (no GPU); full runs: benchmarks/scripts/run_load_experiments.sh
	uv run llmserve sim --sessions 1,2 --turns 2 --max-gen 60 --first-context 1000 2000 --think 0 1 --time-scale 0.05 --out runs/smoke --tag smoke

demo:  ## ~20 s, no GPU: cache thrash, 429 backpressure, crash recovery, /metrics
	uv run python tools/demo.py

report:  ## regenerate docs/15 tables and all plots from benchmarks/results/
	uv run python benchmarks/scripts/load_report.py
	uv run python tools/make_plots.py
