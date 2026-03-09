.PHONY: run up down build-data build-data-docker qa guardrail check-no-direct-data smoke-docker

run:
	streamlit run app/main.py

up:
	docker compose up --build

down:
	docker compose down

build-data:
	python etl/build_data.py --curated-dir data/curated --agg-dir data/agg --duckdb-path analytics.duckdb

build-data-docker:
	docker compose --profile etl run --rm etl

qa:
	pytest -q

guardrail: check-no-direct-data

check-no-direct-data:
	python tools/guardrails/check_no_direct_data_access.py

smoke-docker:
	python qa/docker_smoke.py
