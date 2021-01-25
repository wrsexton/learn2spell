SHELL = bash
UID := $(shell id -u)
GID := $(shell id -g)
PATH_SIG := $(shell echo $$PWD | md5sum | cut -c-8)
DOCKER_DEV_IMAGE = learn2spell:dev-$(PATH_SIG)
DOCKER_RUN_FLAGS = -w "$(PWD)" -v "$(PWD)":"$(PWD)" -e HOME
CURRENT_USER = -u $(UID):$(GID)
DOCKER = docker
DOCKER_RUN = $(DOCKER) run --rm $(CURRENT_USER) $(DOCKER_RUN_FLAGS)

run-dev = exec $(DOCKER_RUN) $1 $(DOCKER_DEV_IMAGE)

all::

clean:
	exec git clean -dfxe .env -e \*.egg-info

run: .setup
	$(call run-dev,$(OPTIONS)) python ./src/learning.py $(ARGS)

check: .setup
	$(call run-dev,$(OPTIONS)) pytest $(ARGS)

.setup: setup.cfg
	exec $(DOCKER) build\
	 --build-arg GID=$(GID)\
	 --build-arg PWD\
	 -t $(DOCKER_DEV_IMAGE)\
	 -f Dockerfile\
	 .
	exec touch '$@'

.PHONY: all clean run check
