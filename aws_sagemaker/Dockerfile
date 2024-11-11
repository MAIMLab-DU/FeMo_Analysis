# Use an official SageMaker base image with Python support
FROM python:3.10-slim

WORKDIR /opt/program

# Install essential packages
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates

COPY requirements.txt /opt/program/requirements.txt
COPY aws_sagemaker/requirements.txt /opt/program/aws_sagemaker/requirements.txt

# Install python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r aws_sagemaker/requirements.txt

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Copy the training script and other files
ADD femo/ /opt/program/femo/
ADD setup.py /opt/program/setup.py
ADD README.md /opt/program/README.md

# Set up the program in the image
RUN pip install .
COPY aws_sagemaker/code /opt/program/

# Make scripts executable
RUN chmod +x /opt/program/train
RUN chmod +x /opt/program/serve
