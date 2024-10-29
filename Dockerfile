FROM python:3.10-slim

WORKDIR /code

COPY requirements.txt .
COPY README.md .
COPY setup.py .
COPY femo/ /code/femo/

RUN pip install -e .

