FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.27 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
ENV PATH="/app/.venv/bin:$PATH"

# Pre-download the embedding / BM25 / reranker models into the image (no network needed at runtime)
RUN python -c "from search.rerank import _encoder; from fastembed import TextEmbedding, SparseTextEmbedding; \
from settings import cfg, ROOT; c=str(ROOT/cfg('search','model_cache')); \
TextEmbedding(cfg('search','dense_model'), cache_dir=c); SparseTextEmbedding(cfg('search','sparse_model'), cache_dir=c); _encoder()"

RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000 8501
CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
