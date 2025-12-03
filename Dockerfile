FROM python:3.12-slim
COPY dupes dupes
COPY requirements.txt requirements.txt
COPY setup.py setup.py
RUN pip install .
CMD uvicorn dupes.api.fast:app --host 0.0.0.0 --port $PORT