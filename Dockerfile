# syntax=docker/dockerfile:1
# Two flavours from one file:
#   docker build -t semcache .                                        # slim: hashing embedder, numpy/faiss index
#   docker build -t semcache:local --build-arg LOCAL_EMBEDDINGS=1 .   # + CPU torch + MiniLM baked in
FROM python:3.12-slim
ARG LOCAL_EMBEDDINGS=0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/opt/hf
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[faiss]" \
 && if [ "$LOCAL_EMBEDDINGS" = "1" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu \
      && pip install ".[local]" \
      && python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')"; \
    fi \
 && useradd --uid 10001 --create-home semcache \
 && mkdir -p /opt/hf && chown -R 10001:10001 /opt/hf
USER 10001:10001
ENV SEMCACHE_EMBEDDER=hashing SEMCACHE_INDEX=auto
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).status == 200 else 1)"
ENTRYPOINT ["semcache"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
