# Contributing

Thanks for looking. A few notes to set expectations right upfront.

## What this is

`korely-graphrag` is the open-source extraction of the retrieval core that
powers [Korely](https://korely.ai). I maintain it as a side project in
my spare time — it's not funded, it doesn't have a roadmap committee, and
there's no SLA on issues or PRs.

It *is* tested (49 tests as of this release) and production-grade
in the sense that Korely commercial runs on the same logic. I'm not going
to ship breaking changes for fun.

## What's welcome

- **Bug reports** — especially if you include a minimal repro and the
  command you ran. For ingestion issues, attach a tiny sample `.md` file
  that triggers the bug. For search/graph issues, attach the query and
  the expected vs actual results.
- **PRs for clear bug fixes** — small, focused, with a test. Read the
  existing tests in `tests/` first to match the style.
- **Docs improvements** — typos, clarifications, better examples. Always
  welcome.
- **New provider implementations** — Ollama is explicitly on the roadmap.
  Implement `BaseProvider` in `src/korely_graphrag/providers/`, wire it in
  `providers/base.py::get_provider`, update `config.py` and the README.

## What's out of scope

- **Major architectural changes** without a discussion first. Please open
  an issue before writing code — I'd rather say "not the direction" in an
  issue than in a PR review.
- **Features that belong in Korely commercial** — auto folder
  classification, intent detection, memory decay, meeting transcription,
  multi-user auth, Stripe billing. See "What's deliberately NOT in this
  repo" in [ARCHITECTURE.md](ARCHITECTURE.md).
- **Dependencies on paid-only models** — the OSS tool must be usable on
  the Gemini free tier (or the upcoming Ollama provider for 100% local).

## Dev setup

```bash
git clone https://github.com/verdana86/korely-graphrag
cd korely-graphrag
cp .env.example .env  # set GEMINI_API_KEY (free tier works)

docker compose up -d db

# Install in editable mode with dev deps
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests (needs the test DB up)
DATABASE_URL='postgresql+psycopg2://korely:korely@localhost:5433/korely_graphrag' \
  pytest tests/
```

Or for tests inside Docker (no local Python needed):

```bash
docker run --rm --network korely-graphrag_default \
  -v $(pwd):/app -w /app \
  -e DATABASE_URL='postgresql+psycopg2://korely:korely@db:5432/korely_graphrag' \
  python:3.12-slim bash -c \
    'pip install -q pydantic pydantic-settings pytest sqlalchemy psycopg2-binary pgvector && PYTHONPATH=src pytest tests/'
```

## Code style

- Python 3.11+ (uses `str | None` and match-style typing)
- Ruff line length 100 (`pyproject.toml`)
- Type hints on public APIs
- No one-liner comments that restate the code. Comment *why*, not *what*.
  See the existing code for the style.

## Commit messages

Single-line summary + optional body explaining the why. Look at the git log
for the style.

## Security

Found a security issue? Please email rather than opening a public issue.
Contact is on [korely.ai](https://korely.ai).
