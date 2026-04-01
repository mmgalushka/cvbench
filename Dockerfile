FROM tensorflow/tensorflow:latest-gpu AS prod

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Copy only what's needed to install the package
COPY pyproject.toml README.md ./
COPY core/ ./core/
COPY cli/ ./cli/
COPY augmentations/ ./augmentations/

# Non-editable install — entry points land in /usr/local/bin
RUN pip install --no-cache-dir . jupyterlab tensorboard

# Pre-create volume mount points so they aren't created as root on first run
RUN mkdir -p /workspace/data /workspace/work_dirs \
             /workspace/augmentations /workspace/configs /workspace/experiments

EXPOSE 8888 6006

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--notebook-dir=/workspace"]

# ── Dev stage — adds test tools and editable reinstall ────────────────────────
FROM prod AS dev

COPY tests/ ./tests/

RUN pip install --no-cache-dir pytest pytest-cov commitizen
RUN pip install --no-cache-dir -e .
