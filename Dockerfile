FROM tensorflow/tensorflow:2.16.2-gpu AS prod

# ── uv (installs into the image's system Python) ──────────────────────────
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /uvx /usr/local/bin/
ENV UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy

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
# Stub the package so uv resolves and caches all dependencies without the real source.
# On code-only changes this layer is reused, skipping the full dependency download.
RUN mkdir -p src/cvbench && touch src/cvbench/__init__.py \
 && uv pip install --no-cache ".[web,export]" jupyterlab \
 && rm -rf src/

COPY src/ ./src/
# Reinstall the package itself — no network I/O, all deps already present
RUN uv pip install --no-cache --no-deps .

# ── Suppress TF banner and ldconfig error from /etc/bash.bashrc ──────────
RUN echo '# cleared by CVBench' > /etc/bash.bashrc

# ── Home directory: volume mount points + shell config ────────────────────
RUN mkdir -p \
      /home/cvbench/data \
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
EXPOSE 8000 8888

# WebUI starts automatically — same pattern as JupyterLab.
# JupyterLab remains available but must be started manually
# inside the container when needed.
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]

# ── Dev stage — adds test tools and editable reinstall ────────────────────
FROM prod AS dev

USER root
WORKDIR /opt/cvbench

COPY tests/ ./tests/

RUN uv pip install --no-cache pytest pytest-cov commitizen
RUN uv pip install --no-cache -e .

USER cvbench
WORKDIR /home/cvbench
