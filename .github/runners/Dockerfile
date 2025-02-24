FROM ghcr.io/actions/actions-runner:latest
USER root

COPY ./.github/runners/runners_dependencies.sh .
RUN ./runners_dependencies.sh

USER runner
CMD /bin/bash
