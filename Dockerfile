FROM python:3.12-slim
COPY dupes dupes
COPY requirements.txt requirements.txt
COPY setup.py setup.py
RUN pip install .

# Environment defaults for model cache/training
ENV MODELS_CACHE_DIR=/models_cache \
    TRAIN_AT_START=true \
    FORCE_RETRAIN=false

RUN mkdir -p /models_cache

COPY scripts scripts
RUN chmod +x scripts/entry.sh

ENTRYPOINT ["scripts/entry.sh"]
CMD ["uvicorn", "dupes.api.fast:app", "--host", "0.0.0.0", "--port", "8000"]
