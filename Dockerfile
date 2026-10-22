# One Dockerfile, two images: build with BACKEND=cuda (NVIDIA) or BACKEND=xpu (Intel Arc).
# Ubuntu 26.04 ships Python 3.14 and the Intel Level Zero packages, so the same base serves both.
FROM ubuntu:26.04
ARG BACKEND=cuda

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
 && if [ "$BACKEND" = xpu ]; then \
      # --no-install-recommends matters: the recommended *-legacy driver makes torch.xpu segfault on Arc 140V.
      apt-get install -y --no-install-recommends libze1 libze-intel-gpu1 intel-opencl-icd; \
    fi \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1

WORKDIR /app
COPY requirements.txt requirements-${BACKEND}.txt ./
RUN pip install --no-cache-dir -r requirements-${BACKEND}.txt

COPY generate.py ./
# models/ and outputs/ are bind-mounted by docker-compose.yml.
ENTRYPOINT ["python", "generate.py"]
