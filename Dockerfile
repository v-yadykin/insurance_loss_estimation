FROM python:3.11-slim AS app

WORKDIR /usr/src/app

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONPATH=/usr/src/app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.0.1
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

RUN useradd -M user
USER user

COPY app .


FROM app AS celery

COPY data /usr/src/data


FROM app AS app_test

USER root

COPY requirements_test.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements_test.txt

COPY tests /usr/src/tests

USER user
