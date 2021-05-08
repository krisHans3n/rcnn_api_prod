# Stage 1: Builder/Compiler
FROM python:3.6-slim as builder
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc

# Stage 2: Runtime
FROM nvidia/cuda:10.1-cudnn7-runtime

# Install Python and dependencies to build pillow and tensorflow
RUN apt-get update \
               && apt-get install -y \
                              python3 \
                              python3-pip \
                              build-essential \
                              python3-dev \
                              zlib1g-dev \
                              libjpeg-dev \
                              wget \
               && pip3 install --upgrade pip 

RUN adduser --system --group --no-create-home app

COPY . /app
WORKDIR /app

COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN apt-get install -y libsm6 libxext6 libxrender-dev

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London
RUN apt-get install -y python3-tk

RUN chown -R app:app /app

EXPOSE 80
USER app

# ENV APP_HOME /APIMain
# WORKDIR $APP_HOME
# COPY . .

# EXPOSE 8080

# RUN chmod +x entrypoint.sh
# ENTRYPOINT ["./entrypoint.sh"]
# CMD ["gunicorn"  , "-b", "0.0.0.0:8000", "APIMain:app"]
