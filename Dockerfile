ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
FROM ${BASE_IMAGE}

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt requirements-cpu.txt ./
ARG REQUIREMENTS=requirements.txt
RUN pip install --no-cache-dir -r ${REQUIREMENTS}

# Copy project
COPY config.yaml .
COPY scripts/ scripts/
COPY data/ data/

# Default: run full pipeline (prepare + finetune + inference)
CMD ["python", "scripts/run_pipeline.py"]
