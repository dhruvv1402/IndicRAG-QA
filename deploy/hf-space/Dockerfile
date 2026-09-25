# IndicRAG-QA on a Hugging Face Docker Space (free CPU tier).
#
# Built from the folder scripts/build-space.py assembles, not from the repo
# root: it carries the code, the web page, the paper files and the frozen
# corpus with its lexical and e5 indices (about 5 MB).

FROM python:3.12-slim

# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    INDICRAG_DATA_DIR=/home/user/app/data \
    INDICRAG_TRUST_PROXY=1 \
    PORT=7860
WORKDIR /home/user/app

# CPU-only torch first: the default wheel bundles ~2.5 GB of CUDA libraries.
COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
RUN pip install --no-cache-dir --user --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir --user .

# Bake the query encoder into the image so the Space starts without a download.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base', device='cpu')"
ENV HF_HUB_OFFLINE=1

COPY --chown=user web ./web
COPY --chown=user paper ./paper
COPY --chown=user docs ./docs
COPY --chown=user data ./data

EXPOSE 7860
# With a GROQ_API_KEY secret set in the Space, answers can come from Groq too;
# without one the Space serves extracted answers only.
CMD ["sh", "-c", "exec indicrag serve --host 0.0.0.0 --port ${PORT} ${GROQ_API_KEY:+--api groq}"]
