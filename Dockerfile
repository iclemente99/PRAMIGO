FROM ubuntu:24.04

LABEL org.opencontainers.image.description="BiomiX PRAMIGO image - Heterogeneous Graph Transformer multi-omic integration"

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      ca-certificates \
      build-essential \
      git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (same tool used in the README's manual setup)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /opt/pramigo

# Clone PRAMIGO directly from GitHub (the repository is public).
# PRAMIGO_REF can be a branch, a tag or - best for reproducible builds -
# a commit hash. Override at build time, e.g.:
#   docker build --build-arg PRAMIGO_REF=<commit-hash> -t biomix-pramigo:test .
# Note: Docker caches this step. If PRAMIGO_REF is a branch name and the
# branch gets new commits, rebuild with --no-cache (or pass a commit hash).
ARG PRAMIGO_REPO=https://github.com/iclemente99/PRAMIGO.git
ARG PRAMIGO_REF=Implementation_Docker
RUN git init . \
 && git remote add origin "${PRAMIGO_REPO}" \
 && git fetch --depth 1 origin "${PRAMIGO_REF}" \
 && git checkout FETCH_HEAD

# uv downloads its own Python 3.8 build if needed - independent of the OS's
# default Python - then installs the pinned dependencies into that venv.
#
# torch build depends on the target architecture (set automatically by
# docker buildx / BuildKit):
#   - amd64 (Windows/Linux, NVIDIA GPU possible): CUDA 11.7 build of torch.
#   - arm64 (e.g. Apple Silicon Macs): no CUDA build exists, so the standard
#     CPU-only torch from PyPI is used. PRAMIGO then runs on CPU (--cuda 1).
ARG TARGETARCH
RUN uv venv hgt_env --python 3.8 \
 && if [ "${TARGETARCH}" = "arm64" ]; then \
      uv pip install --python hgt_env/bin/python torch==1.13.1 ; \
    else \
      uv pip install --python hgt_env/bin/python --index-url https://download.pytorch.org/whl/cu117 torch==1.13.1 ; \
    fi \
 && uv pip install --python hgt_env/bin/python -r env/biomixhgt_env.txt

# So `python`, `pytest`, etc. resolve to the venv by default when you exec
# into the container or override the entrypoint.
ENV PATH="/opt/pramigo/hgt_env/bin:${PATH}"
