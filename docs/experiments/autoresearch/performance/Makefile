ENV_FILE ?= .env

.PHONY: bootstrap create-run iteration loop install-upstream

bootstrap:
	./scripts/bootstrap.sh $(ENV_FILE)

install-upstream:
	./scripts/install_upstream_autoresearch.sh $(ENV_FILE)

create-run:
	./scripts/create_run.sh $(ENV_FILE)

iteration:
	./scripts/run_iteration.sh $(ENV_FILE)

loop:
	./scripts/run_loop.sh $(ENV_FILE)
