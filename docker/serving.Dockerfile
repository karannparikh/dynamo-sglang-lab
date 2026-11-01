# syntax=docker/dockerfile:1.7

FROM rust:1.88.0-bookworm AS dynamo-builder

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        clang \
        cmake \
        libclang-dev \
        libprotobuf-dev \
        patchelf \
        pkg-config \
        protobuf-compiler \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Debian ships the well-known *.proto types (google/protobuf/struct.proto, used
# by the vllm sidecar) in libprotobuf-dev under /usr/include; point prost/tonic
# at it so from-source Dynamo builds can resolve the imports.
ENV PROTOC_INCLUDE=/usr/include

RUN python3 -m venv /build-venv \
    && /build-venv/bin/pip install --no-cache-dir \
        build==1.2.2.post1 \
        hatchling==1.27.0 \
        maturin==1.9.3 \
        pip==25.2

COPY --from=dynamo / /src/dynamo

RUN test -f /src/dynamo/pyproject.toml \
    || { echo "DYNAMO_SOURCE does not contain pyproject.toml" >&2; exit 1; }
RUN test -f /src/dynamo/lib/bindings/python/Cargo.toml \
    || { echo "DYNAMO_SOURCE cannot build ai-dynamo-runtime: lib/bindings/python/Cargo.toml is missing" >&2; exit 1; }

RUN mkdir -p /wheels \
    && cd /src/dynamo \
    && /build-venv/bin/python -m build --wheel --outdir /wheels

RUN cd /src/dynamo/lib/bindings/python \
    && FEATURES="" \
    && for feature in kv-indexer slot-tracker select-service; do \
         if grep -q "^${feature}[[:space:]]*=" Cargo.toml; then \
           FEATURES="${FEATURES}${FEATURES:+,}${feature}"; \
         fi; \
       done \
    && if [ -n "${FEATURES}" ]; then \
         /build-venv/bin/maturin build --release --features "${FEATURES}" --out /wheels; \
       else \
         /build-venv/bin/maturin build --release --out /wheels; \
       fi \
    || { echo "failed to build ai-dynamo-runtime from the selected DYNAMO_SOURCE" >&2; exit 1; }

FROM python:3.11-slim-bookworm AS serving

ARG DYNAMO_COMMIT
ARG FAKE_SGLANG_COMMIT

LABEL org.opencontainers.image.title="Dynamo routing harness serving runtime" \
      io.dynamo-routing-harness.dynamo-commit="${DYNAMO_COMMIT}" \
      io.dynamo-routing-harness.fake-sglang-commit="${FAKE_SGLANG_COMMIT}"

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/runtime-requirements.txt /tmp/runtime-requirements.txt
COPY docker/dynamo-base-requirements.txt /tmp/dynamo-base-requirements.txt
RUN python -m pip install --no-cache-dir pip==25.2 \
    && python -m pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --requirement /tmp/runtime-requirements.txt \
    && python -m pip install --no-cache-dir \
        --requirement /tmp/dynamo-base-requirements.txt

COPY --from=dynamo-builder /wheels /opt/dynamo/wheels
RUN python -m pip install --no-cache-dir --no-deps \
        /opt/dynamo/wheels/ai_dynamo_runtime*.whl \
        /opt/dynamo/wheels/ai_dynamo-*.whl

COPY --from=fake_sglang / /opt/fake-sglang
RUN test -f /opt/fake-sglang/pyproject.toml \
    || { echo "FAKE_SGLANG_SOURCE does not contain pyproject.toml" >&2; exit 1; } \
    && python -m pip install --no-cache-dir --no-deps --editable /opt/fake-sglang

COPY docker/validate_contract.py /opt/harness/validate_contract.py
RUN mkdir -p /opt/harness \
    && python /opt/harness/validate_contract.py --output /opt/harness/effective-profile.json \
    && python -m dynamo.frontend --help >/dev/null \
    && python -m dynamo.sglang --help >/dev/null \
    || { echo "selected local Dynamo/fake-sglang sources do not satisfy the harness runtime contract" >&2; exit 1; }

RUN useradd --create-home --uid 1000 --shell /bin/bash simulator \
    && chown -R simulator:simulator /opt/harness /opt/fake-sglang

ENV DYNAMO_SGLANG_PATH=/usr/local/lib/python3.11/site-packages/dynamo/sglang \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

WORKDIR /opt/fake-sglang
USER simulator

CMD ["python", "-m", "dynamo.sglang", "--help"]

FROM python:3.12-slim-bookworm AS planner

ARG DYNAMO_COMMIT
LABEL org.opencontainers.image.title="Dynamo routing harness Planner" \
      io.dynamo-routing-harness.dynamo-commit="${DYNAMO_COMMIT}"

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=dynamo /container/deps/requirements.planner.txt /tmp/requirements.planner.txt
RUN test -s /tmp/requirements.planner.txt \
    || { echo "DYNAMO_SOURCE cannot supply the Planner runtime: container/deps/requirements.planner.txt is missing or empty" >&2; exit 1; } \
    && python -m pip install --no-cache-dir pip==25.2 \
    && python -m pip install --no-cache-dir --requirement /tmp/requirements.planner.txt

COPY --from=dynamo-builder /wheels /opt/dynamo/wheels
RUN python -m pip install --no-cache-dir /opt/dynamo/wheels/ai_dynamo_runtime*.whl /opt/dynamo/wheels/ai_dynamo-*.whl \
    && python -m dynamo.planner --help >/dev/null \
    || { echo "selected DYNAMO_SOURCE does not provide a working local Planner runtime" >&2; exit 1; }

RUN useradd --create-home --uid 1000 --shell /bin/bash dynamo
USER dynamo
WORKDIR /home/dynamo
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "dynamo.planner", "--help"]
