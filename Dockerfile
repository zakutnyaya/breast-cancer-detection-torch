FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt install -y git git-lfs ffmpeg curl wget unzip zip

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

WORKDIR /app

RUN pip install "poetry>=1.4.1"
RUN poetry config virtualenvs.create false

ADD pyproject.toml poetry.lock /app/
RUN poetry install --no-root --only main && rm -rf ~/.cache

RUN conda install pytorch==1.11.0 torchvision==0.12.0 pytorch-cuda -c pytorch -c nvidia
RUN git clone https://github.com/ultralytics/yolov5.git
RUN pip install timm psutil matplotlib seaborn wandb torchmetrics

ADD . /app/
ENV PYTHONPATH /app/