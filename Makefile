.PHONY: install check lint test e2e mutation bench docker
install:
	pip install -e ".[dev,faiss]"
lint:
	ruff check .
	ruff format --check src tests bench scripts
	mypy
test:
	python -m pytest -q -m "not slow"
e2e:
	python -m pytest -q -m slow
check: lint test
mutation:
	scripts/mutation.sh && python scripts/mutation_summary.py
bench:
	python bench/make_workload.py && python bench/sweep.py && bench/run_loadtest.sh
docker:
	docker build -t semcache .
