FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu18.04
ARG PYTHON_VERSION=3.8

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*


RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH
# This must be done before pip so that requirements.txt is available
WORKDIR /workspace
COPY . .
# Install requirements
RUN pip install -r requirements.txt
# Install PyTorch
RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

RUN chmod -R a+w .