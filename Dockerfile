# Builder
FROM cgr.dev/chainguard/python:latest-dev AS builder
USER root
RUN install -d -o nonroot -g nonroot /opt/venv \
    && mkdir -p /workspace && chown -R nonroot:nonroot /workspace
USER nonroot
ENV VENV=/opt/venv PATH="/opt/venv/bin:$PATH"
WORKDIR /workspace
RUN python -m venv "$VENV"
COPY requirements.txt .
RUN pip install --upgrade pip wheel setuptools && pip install -r requirements.txt
COPY app ./app
RUN mkdir -p /workspace/data/chroma /workspace/data/docs

# Runtime
FROM cgr.dev/chainguard/python:latest
ENV VENV=/opt/venv PATH="/opt/venv/bin:$PATH"
WORKDIR /workspace
COPY --from=builder /opt/venv /opt/venv
COPY --chown=nonroot:nonroot app ./app
USER nonroot

# Match how you actually start the server
ENTRYPOINT []
CMD ["/opt/venv/bin/uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--reload"]