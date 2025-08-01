# Use an official SageMaker base image with Python support
FROM public.ecr.aws/docker/library/python:3.10-slim

WORKDIR /opt/program

# Install essential packages including build tools for C++ compilation
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
         build-essential \
         g++ \
         cmake \
         && rm -rf /var/lib/apt/lists/*

# Copy the pyproject.toml and femo package files
COPY pyproject.toml /opt/program/pyproject.toml
COPY setup.py /opt/program/setup.py
COPY femo/ /opt/program/femo/

# Install python dependencies using pip (via pyproject.toml)
RUN pip install --upgrade pip
RUN pip install .[sagemaker]

# Set environment variables for Python
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

ADD README.md /opt/program/README.md
COPY aws_sagemaker/code /opt/program/

# Make scripts executable
RUN chmod +x /opt/program/train
RUN chmod +x /opt/program/serve