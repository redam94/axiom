.PHONY: install tests fast_tests format format_check lint types gates notebooks docs docs_publish examples benchmarks

install:
	uv sync --group dev

tests:
	uv run pytest tests/ -n logical --cov=axiom --cov-report=term-missing

fast_tests:
	uv run pytest tests/ -n logical -m 'not slow'

format:
	uv run black src tests examples benchmarks

format_check:
	uv run black --check --diff src tests examples benchmarks

lint:
	uv run ruff check src tests examples benchmarks

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

# Sphinx API reference; -W makes every warning fatal so the reference stays
# warning-free. Output lands in docs/_build/html (git-ignored). `-d` keeps the
# doctree cache beside the HTML rather than inside it: in it, the tree is 270 MB
# of pickles wrapped around 23 MB of pages, which is what Read the Docs would
# have uploaded.
docs:
	uv run --group docs sphinx-build -W --keep-going -b html \
		-d docs/_build/doctrees docs docs/_build/html

# Ask Read the Docs to build now, rather than waiting for a push to be noticed.
# Needs a token from https://readthedocs.org/accounts/tokens/:
#
#     export READTHEDOCS_TOKEN=...
#     make docs_publish                 # builds the `latest` version
#     make docs_publish VERSION=stable  # or a named one
#
# The routine path is a push: Read the Docs pulls from the repository and builds
# from .readthedocs.yaml on its own. This is for when you want it now.
docs_publish:
	@test -n "$$READTHEDOCS_TOKEN" || { \
		echo "set READTHEDOCS_TOKEN (https://readthedocs.org/accounts/tokens/)"; exit 1; }
	@curl -fsS -X POST \
		-H "Authorization: Token $$READTHEDOCS_TOKEN" \
		"https://readthedocs.org/api/v3/projects/$${SLUG:-axiom}/versions/$${VERSION:-latest}/builds/" \
		| python3 -c "import json,sys; b=json.load(sys.stdin)['build']; \
		print(f\"build {b['id']} {b['state']}: {b['urls']['build']}\")"

# Every example under examples/, one per field. They run on the core install
# with no extras, and the site's examples page is generated from their output —
# so a broken example is a broken page.
examples:
	uv run python examples/run_all.py

# Real datasets with published results: does axiom reproduce numbers other people
# computed, from their raw data? Vendored datasets run anywhere; the two whose
# licence forbids redistribution skip until `python benchmarks/fetch.py --all`.
benchmarks:
	uv run pytest tests/benchmarks -q
	uv run python benchmarks/run_all.py
