FROM ghcr.io/actions/actions-runner:latest

COPY ./.github/runners/runners_dependencies.sh .
RUN ./runners_dependencies.sh

CMD /bin/bash
