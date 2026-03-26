# Contributing

## Setup

```bash
uv sync --extra dev --extra scheduler
```

## Running tests

```bash
uv run pytest tests/
```

With coverage report:

```bash
uv run pytest tests/ --cov=eostrata --cov-report=term-missing
```

## Linting and formatting

Check for code quality issues:

```bash
uv run ruff check eostrata/ tests/
```

Auto-fix what can be fixed automatically:

```bash
uv run ruff check eostrata/ tests/ --fix
```

Apply consistent formatting:

```bash
uv run ruff format eostrata/ tests/
```

Check formatting without modifying files (useful in CI):

```bash
uv run ruff format --check eostrata/ tests/
```

## CI

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs linting and tests automatically on every push and pull request.
