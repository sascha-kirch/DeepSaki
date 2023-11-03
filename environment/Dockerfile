ARG TF_VERSION="2.14.0-gpu"
FROM tensorflow/tensorflow:${TF_VERSION}

# change python version to 3.10. Image comes with 3.10 and 3.11, but 3.11 is set to default
ARG PYTHON_VERSION="3.10"

WORKDIR /usr/bin
RUN unlink python && \
    ln -s /usr/bin/python${PYTHON_VERSION} python
WORKDIR ~

# add requirements file
ADD requirements.txt .
ADD requirements-dev.txt .

# install packages and requirements
RUN apt update && \
    apt install python3-pip -y && \
    python -m pip install --upgrade pip && \
    apt install ffmpeg libsm6 libxext6  -y && \
    apt install graphviz -y && \
    apt install dos2unix -y && \
    apt install git -y && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt

WORKDIR /deepsaki
