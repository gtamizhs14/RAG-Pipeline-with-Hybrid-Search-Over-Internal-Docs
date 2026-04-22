FROM python:3.11-slim

WORKDIR /app

# gcc is required to build rank-bm25 and some sentence-transformers wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding and reranker models at build time so containers
# start instantly without waiting for a HuggingFace download on first request.
# The cache lands in /root/.cache/huggingface inside the image layer.
RUN python - <<'EOF'
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer("all-MiniLM-L6-v2")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
EOF

COPY src/ ./src/
COPY streamlit_app.py .
COPY scripts/ ./scripts/

ENV PYTHONPATH=/app

# Default: run the FastAPI server. Override CMD in docker-compose for Streamlit.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
