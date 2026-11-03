.PHONY: setup test lint eval extract qa-log compare score demo serve ui docker

PY := uv run python

setup:            ## install pinned dependencies (uv.lock) and create .env from the template
	uv sync
	@test -f .env || cp .env.example .env

lint:
	uv run ruff check .
	uv run mypy ingestion search agents api settings.py

test:
	uv run pytest -q --cov=ingestion --cov=search --cov=agents --cov=api --cov-report=term-missing

eval:             ## retrieval evaluation -> eval/report.md
	$(PY) main.py eval --chunk-sizes 1000,2000,3000

extract:          ## extraction mode for both bids -> outputs/Bid1.json, outputs/Bid2.json
	$(PY) main.py --bid ./Bid1
	$(PY) main.py --bid ./Bid2

qa-log:
	$(PY) main.py qa-log

compare:
	$(PY) main.py compare

score:            ## extraction accuracy vs eval/extraction_gold.json
	$(PY) -m eval.score_extraction

demo: extract qa-log compare score   ## everything shown in the demo video, in order
	$(PY) main.py ask "What is the submission deadline for Bid1 after all addendums?"

serve:
	$(PY) main.py serve

ui:
	$(PY) main.py ui

docker:
	docker compose up --build
