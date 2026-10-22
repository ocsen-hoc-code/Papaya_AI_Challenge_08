# Shortcuts for common tasks. Run `make help` to list them.

BIN := .venv/bin

.PHONY: help setup test test-slow lint run eval clean-venv

help:  ## list available commands
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-10s %s\n", $$1, $$2}'

setup:  ## create .venv and install everything (auto-detects Python 3.10+ and hardware)
	./setup.sh

test:  ## unit tests (no model needed, ~5 s)
	$(BIN)/pytest -q

test-slow:  ## tests that run the real model (~2 min)
	$(BIN)/pytest -m slow -q

lint:  ## ruff lint
	$(BIN)/ruff check src tests scripts

run:  ## extract one or more files: make run FILE=path/to/doc.pdf  (add OUT=dir to save JSON files)
	@test -n "$(FILE)" || { echo "usage: make run FILE=path/to/document.pdf [OUT=output/]"; exit 2; }
	$(BIN)/medextract run $(FILE) $(if $(OUT),-o $(OUT))

eval:  ## run the 10 test documents and score them against ground_truth/ (~25 min)
	$(BIN)/medextract run document-tests/ -o results/
	$(BIN)/python scripts/eval.py

clean-venv:  ## delete .venv (run `make setup` to rebuild)
	rm -rf .venv
