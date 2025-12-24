# Default target
default: lint

lint:
	uv run -m black .
	uv run ruff check --fix
	uv run -m mypy .

format:
	uv run -m isort .
	uv run -m black .
