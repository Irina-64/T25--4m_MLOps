FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

COPY requirements.txt pyproject.toml .python-version uv.lock ./

RUN uv venv

RUN uv sync --group docker

COPY .. .

FROM python:3.12-slim-bookworm AS final

WORKDIR /app

# Копируем виртуальное окружение из этапа builder
COPY --from=builder /app/.venv /app/.venv

# Копируем код приложения
COPY --from=builder /app /app

# Устанавливаем пути Python для использования виртуального окружения
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

CMD ["python", "src/api.py"]
