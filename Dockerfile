FROM ubuntu:20.04

RUN apt-get update -qq -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    	build-essential \
        wget \
        python3-dev \
        python3-pip \
        git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /finnessayscore/

ENV LANG='C.UTF-8' LC_ALL='C.UTF-8'

RUN python3 -m pip install --upgrade poetry==1.1.7

ADD pyproject.toml poetry.lock /finnessayscore/

RUN poetry export \
      --without-hashes > requirements.txt && \
    sed -i '/pytorch/d' requirements.txt && \
    python3 -m pip install -r requirements.txt && \
    rm requirements.txt && \
    pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install pytorch-lightning && \
    rm -rf /root/.cache

RUN echo "/finnessayscore" > \
    /usr/local/lib/python3.8/dist-packages/finnessayscore.pth

RUN ln -sf /usr/bin/python3 /usr/bin/python

ADD . /finnessayscore/
