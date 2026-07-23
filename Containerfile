FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY container-requirements.txt /tmp/container-requirements.txt
RUN pip install --no-cache-dir -r /tmp/container-requirements.txt

CMD ["sleep", "infinity"]
