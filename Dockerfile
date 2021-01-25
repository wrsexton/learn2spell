#syntax=docker/dockerfile:1.2
FROM python:slim
RUN apt-get -y update\
 && apt-get -y install gosu
COPY setup.* /tmp/
RUN --mount=type=cache,target=/root/.cache\
 cd /tmp\
 && pip install -e '.[dev]'\
 && pip uninstall -y $(./setup.py --name)
ARG GID
ARG PWD
WORKDIR $PWD
RUN --mount=type=cache,target=/root/.cache\
 --mount=type=bind,target=.,rw\
 gosu 0:$GID pip install -e '.[dev]'
EXPOSE 80
