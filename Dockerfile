# Dockerfile for BDSRAE Topic 19: Subgroup Performance in Healthcare Risk Models
# Author: Ankith Umadi, Haneesh E, Vikas R (PES University)

FROM python:3.12-slim

# Install OpenJDK-17 for Apache PySpark, procps for memory monitoring, and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    procps \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

WORKDIR /workspace

# Copy requirements and install
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source repository
COPY . /workspace/

EXPOSE 8888

# Default to running the full pipeline
CMD ["python", "run_pipeline.py", "--cohort_size", "25000"]
