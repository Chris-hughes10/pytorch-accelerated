# Builds GPU docker image
# Uses multi-staged approach to reduce size
FROM condaforge/mambaforge:latest AS compile-image

ENV PYTHON_VERSION=3.10.6

# Create our conda env
RUN conda create --name core python=${PYTHON_VERSION}

# Make bash the default shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Stage 2
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04 AS build-image
COPY --from=compile-image /opt/conda /opt/conda
ENV PATH /opt/conda/bin:$PATH

RUN echo "source activate core" >> ~/.profile

# COPY requirements.txt .

# RUN pip install -r requirements.txt