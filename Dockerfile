FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# build-essential is needed to build sentencepiece/tokenizers wheels on some
# platforms; not removed afterwards since torch/transformers also need libgomp
# etc. at runtime and the slim base is already small enough.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch wheel — the default PyPI wheel pulls in CUDA libraries
# (multiple GB) that this app never uses, since inference runs on CPU.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2" \
    && pip install -r requirements.txt \
    && python -m spacy download ru_core_news_lg

COPY . .

# Pre-download the Kazakh NER transformer at build time so the image works
# offline and the first user request doesn't pay the ~1 min cold-start cost.
RUN python -c "\
from recognizers import KK_TRANSFORMER_MODEL; \
from transformers import AutoModelForTokenClassification, AutoTokenizer; \
AutoTokenizer.from_pretrained(KK_TRANSFORMER_MODEL); \
AutoModelForTokenClassification.from_pretrained(KK_TRANSFORMER_MODEL)"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py"]
