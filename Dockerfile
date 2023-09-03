FROM python:3.10.12-slim as runtime

ENV PIP_ROOT_USER_ACTION=ignore
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install --assume-yes git

WORKDIR /usr/src/app
#COPY . .
RUN git clone https://github.com/timekeeper18/itmo_reliableml.git
WORKDIR /usr/src/app/itmo_reliableml

ENV PYTHONOPTIMIZE true
ENV DEBIAN_FRONTEND noninteractive

# setup timezone
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN pip install -U --no-cache-dir pip poetry setuptools wheel
RUN poetry install --no-root

CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0"]
