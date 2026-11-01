# Prebuilt vanilla serving runtime, layered with fake-sglang. Python 3.10 keeps
# Dynamo 1.6.0 on its lightweight dependency path without AISimulate.
FROM python:3.10-slim-bookworm AS serving

ARG DYNAMO_RUNTIME_VERSION
ARG FAKE_SGLANG_COMMIT

LABEL org.opencontainers.image.title="Dynamo routing harness serving runtime (prebuilt)" \
      io.dynamo-routing-harness.dynamo-version="${DYNAMO_RUNTIME_VERSION}" \
      io.dynamo-routing-harness.fake-sglang-commit="${FAKE_SGLANG_COMMIT}"

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/runtime-requirements.txt /tmp/runtime-requirements.txt
RUN python -m pip install --no-cache-dir pip==25.2 \
    && python -m pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --requirement /tmp/runtime-requirements.txt \
    && python -m pip install --no-cache-dir \
        "ai-dynamo==${DYNAMO_RUNTIME_VERSION}"

COPY --from=fake_sglang / /opt/fake-sglang
RUN test -f /opt/fake-sglang/pyproject.toml \
    || { echo "FAKE_SGLANG_SOURCE does not contain pyproject.toml" >&2; exit 1; } \
    && python -m pip install --no-cache-dir --no-deps --editable /opt/fake-sglang

COPY docker/validate_contract.py /opt/harness/validate_contract.py
RUN mkdir -p /opt/harness \
    && python /opt/harness/validate_contract.py --output /opt/harness/effective-profile.json \
    && python -c "import sglang; assert sglang.__is_fake_sglang__" \
    && python -m dynamo.frontend --help >/dev/null \
    && python -m dynamo.sglang --help >/dev/null \
    || { echo "prebuilt ai-dynamo and the selected fake-sglang source do not satisfy the harness runtime contract" >&2; exit 1; }

RUN useradd --create-home --uid 1000 --shell /bin/bash simulator \
    && chown -R simulator:simulator /opt/harness /opt/fake-sglang

ENV DYNAMO_SGLANG_PATH=/usr/local/lib/python3.10/site-packages/dynamo/sglang \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

WORKDIR /opt/fake-sglang
USER simulator

CMD ["python", "-m", "dynamo.sglang", "--help"]
