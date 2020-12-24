UID := $(shell id -u)
GID := $(shell id -g)
PATH_SIG := $(shell echo $$PWD | md5sum | cut -c-8)
DOCKER_DEV_IMAGE = learn2spell-dev:$(PATH_SIG)
DOCKER_RUN_FLAGS = -w "$(PWD)" -v "$(PWD)":"$(PWD)" -e HOME
CURRENT_USER = -u $(UID):$(GID)
DOCKER = docker
DOCKER_RUN = $(DOCKER) run --rm $(CURRENT_USER) $(DOCKER_RUN_FLAGS)

run-dev = $(DOCKER_RUN) $1 $(DOCKER_DEV_IMAGE) $2

all::

clean:
	exec git clean -dfxe .env -e \*.egg-info
run: .setup
	exec $(call run-dev,$(OPTIONS),python ./src/learning.py $(ARGS))

check: .setup
	exec $(call run-dev,$(OPTIONS),pytest $(ARGS))

.setup: setup.cfg
	if [ -f .setup ]; then rm .setup; fi
	exec bash -c "\
	set -e;\
	$(DOCKER) build\
	 --build-arg GID=$(GID)\
	 --build-arg PWD\
	 -t $(DOCKER_DEV_IMAGE)\
	 -f Dockerfile\
	 .;\
	: >.setup"

.PHONY: all clean check
