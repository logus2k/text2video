# Ubuntu 26.04 ships Python 3.14; the torch wheel brings its own CUDA runtime, so no nvidia/cuda base image is needed.
FROM ubuntu:26.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY generate.py chain.py ./
# models/ and outputs/ are bind-mounted by docker-compose.yml.
ENTRYPOINT ["python", "generate.py"]
