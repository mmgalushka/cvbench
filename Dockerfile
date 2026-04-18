FROM tensorflow/tensorflow:latest-gpu AS prod

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user ─────────────────────────────────────────────────────────
RUN groupadd --gid 1000 cvbench \
 && useradd  --uid 1000 --gid 1000 --create-home --shell /bin/bash cvbench

# ── Suppress TF C++ logging (GPU library warnings, etc.) ─────────────────
ENV TF_CPP_MIN_LOG_LEVEL=3

# ── Install package into system Python ────────────────────────────────────
WORKDIR /opt/cvbench

COPY pyproject.toml README.md ./
COPY src/ ./src/

# Non-editable install — entry points land in /usr/local/bin
# cvbench[web] adds fastapi + uvicorn for the WebUI (served by `serve` command)
RUN pip install --no-cache-dir ".[web]" jupyterlab tensorboard

# ── Suppress TF banner and ldconfig error from /etc/bash.bashrc ──────────
RUN echo '# cleared by CVBench' > /etc/bash.bashrc

# ── Home directory: volume mount points + shell config ────────────────────
RUN mkdir -p \
      /home/cvbench/data \
      /home/cvbench/augmentations \
      /home/cvbench/workspace \
      /home/cvbench/experiments \
 && chown -R cvbench:cvbench /home/cvbench

COPY scripts/bashrc /home/cvbench/.bashrc
RUN chown cvbench:cvbench /home/cvbench/.bashrc

# ── Runtime ───────────────────────────────────────────────────────────────
USER cvbench
WORKDIR /home/cvbench

# 8000 — CVBench WebUI (serve)
# 8888 — JupyterLab (start manually: jupyter lab --ip=0.0.0.0 --no-browser)
# 6006 — TensorBoard (start manually: tensorboard --logdir experiments/)
EXPOSE 8000 8888 6006

# WebUI starts automatically — same pattern as JupyterLab / TensorBoard.
# JupyterLab and TensorBoard remain available but must be started manually
# inside the container when needed.
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]

# ── Dev stage — adds test tools and editable reinstall ────────────────────
FROM prod AS dev

USER root
WORKDIR /opt/cvbench

COPY tests/ ./tests/

RUN pip install --no-cache-dir pytest pytest-cov commitizen
RUN pip install --no-cache-dir -e .

USER cvbench
WORKDIR /home/cvbench
