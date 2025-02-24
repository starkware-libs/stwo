FROM ghcr.io/actions/actions-runner:latest

USER runner
COPY runners_dependencies.sh .
RUN ./runners_dependencies.sh

CMD /bin/bash
