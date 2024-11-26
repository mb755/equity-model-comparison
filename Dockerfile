FROM continuumio/miniconda3

RUN mkdir -p equity-model-comparison

COPY . /equity-model-comparison
WORKDIR /equity-model-comparison

RUN apt-get update && apt-get install -y doxygen graphviz git

RUN conda env create --name equity-model-comparison --file environment.yml

RUN echo "conda activate equity-model-comparison" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN pre-commit install
