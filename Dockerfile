FROM ubuntu:16.04

RUN apt-get update

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update

RUN apt install -y build-essential libstdc++6 wget git swig libssl-dev python3.6 python3.6-dev python3-pip python3.6-venv locales


RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL en_US.UTF-8

RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel nltk 

ADD . /app

WORKDIR /app
VOLUME /app/classification 