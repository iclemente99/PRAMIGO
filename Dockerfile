FROM ubuntu:24.04

LABEL org.opencontainers.image.description="BiomiX PRAMIGO image - Heterogeneous Graph Transformer multi-omic integration"

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      ca-certificates \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (same tool used in the README's manual setup)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /opt/pramigo

# For now: copy the local repo directly (later this becomes a git clone).
COPY . .

# uv downloads its own Python 3.8 build if needed - independent of the OS's
# default Python - then installs the pinned dependencies into that venv.
RUN uv venv hgt_env --python 3.8 && uv pip install --python hgt_env/bin/python --index-url https://download.pytorch.org/whl/cu117 torch==1.13.1 && uv pip install --python hgt_env/bin/python -r env/biomixhgt_env.txt

# So `python`, `pytest`, etc. resolve to the venv by default when you exec
# into the container or override the entrypoint.
ENV PATH="/opt/pramigo/hgt_env/bin:${PATH}"
