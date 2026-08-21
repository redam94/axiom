.PHONY: install tests fast_tests format format_check lint types gates notebooks

install:
	uv sync --group dev

tests:
	uv run pytest tests/ -n logical --cov=axiom --cov-report=term-missing

fast_tests:
	uv run pytest tests/ -n logical -m 'not slow'

format:
	uv run black src tests examples

format_check:
	uv run black --check --diff src tests examples

lint:
	uv run ruff check src tests examples

types:
	uv run mypy

# The twelve gates that encode the repo's design decisions. See
# docs/plan/04-contracts-and-testing.md.
gates:
	uv run pytest tests/contracts -q

# Execute every notebook series under nbs/. Each subpackage's series is a
# phase deliverable; gate 12 checks that every public symbol appears in one.
notebooks:
	uv run pytest --nbmake nbs/ -n logical
