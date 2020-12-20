FROM python:3.7-slim
RUN apt-get -y update\
    && apt-get -y install gosu
ARG GID
ARG PWD
WORKDIR $PWD
RUN --mount=type=bind,target=.,rw gosu 0:$GID pip install -e '.[dev]'
EXPOSE 80