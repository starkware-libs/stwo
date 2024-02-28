FROM ghcr.io/actions/actions-runner:latest
USER root

RUN apt update && \
    apt install -y \
    build-essential \
    curl

USER runner
CMD /bin/bash
