FROM tensorflow/tensorflow:2.14.0-gpu

ADD requirements.txt .
ADD requirements-dev.txt .

RUN apt update && \
    apt install ffmpeg libsm6 libxext6  -y && \
    apt install graphviz -y && \
    apt install dos2unix && \
    apt install git && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt