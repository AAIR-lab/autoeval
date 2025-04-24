FROM ubuntu:jammy
LABEL authors="Rushang Karia"
WORKDIR /autoeval

COPY ./ ./

ENV PYTHONHASHSEED=0

RUN ./scripts/install.sh
RUN rm dataset.zip
RUN rm -fr .git*