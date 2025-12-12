FROM python:3.12-slim
COPY requirements.txt requirements.txt
COPY setup.py setup.py
RUN pip install .
COPY dupes dupes
COPY gcp gcp/
RUN pip install --no-cache-dir sentence-transformers torch
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"
CMD uvicorn dupes.api.fast:app --host=0.0.0.0 --port=$PORT
