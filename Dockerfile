# Base image with tensorflow and all the dependence of 4Dsurvival
# lisurui6/4dsurvival-gpu:1.1

FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN apt-get update && apt-get install -y build-essential git libjpeg-dev && \
    apt-get install -y vim tmux curl

RUN pip install --upgrade pip setuptools && pip install tensorflow && \
    pip install --upgrade keras==2.2.2 lifelines==0.23.9 optunity matplotlib sklearn scipy pandas numpy pyhocon

WORKDIR /root


COPY . /root/4Dsurvival

RUN cd 4Dsurvival && python setup.py develop
