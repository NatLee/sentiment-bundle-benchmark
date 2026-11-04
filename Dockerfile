# GPU image for the full benchmark. The base is shared with other PyTorch projects on the same
# machine, so its layers are only stored once.
ARG BASE_IMAGE=pytorch/pytorch:2.14.1-cuda13.2-cudnn9-runtime
FROM ${BASE_IMAGE}

WORKDIR /app
# The base image puts torch in the system Python and ships no venv module, so we add to it the same way.
ENV PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HOME=/cache/huggingface

COPY constraints.txt pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir -c constraints.txt ".[models,test]"

COPY configs ./configs
COPY tests ./tests
ENTRYPOINT ["sbb"]
CMD ["--help"]
