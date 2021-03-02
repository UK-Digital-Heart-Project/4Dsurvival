FROM tensorflow/tensorflow:1.10.1-devel-gpu-py3

RUN apt-get update && apt-get install -y build-essential git libjpeg-dev && \
    apt-get install -y vim tmux curl

WORKDIR /root

RUN git clone https://github.com/UK-Digital-Heart-Project/4Dsurvival.git && cd 4Dsurvival && git fetch \
    && pip3 install --upgrade pip setuptools && python setup.py develop

RUN pip3 install --upgrade keras==2.2.2 lifelines optunity matplotlib sklearn scipy pandas numpy
