ARG TAG
ARG UBUNTU_VERSION
ARG CUDA_IMAGE_VERSION

ARG BASE_IMAGE=nvidia/cuda:${CUDA_IMAGE_VERSION}-${TAG}-ubuntu${UBUNTU_VERSION}
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

LABEL maintainer="hugo.pechman@outlook.cz" \
      org.label-schema.schema-version="1.0.0" \
      org.label-schema.name="nvidia-cuda-image" \
      org.label-schema.description="Image for LLM model development." \
      org.label-schema.url="https://github.com/petrpechman/czech_gec"

RUN apt-get update && apt-get install -y build-essential pkg-config curl \
    software-properties-common unzip perl libtool gettext autoconf automake \
    texinfo autopoint git vim wget

# install Python:
ARG PYTHON_VERSION=python3.10
COPY docker-data/setup_python.sh /setup_python.sh
RUN chmod 755 /setup_python.sh
RUN /setup_python.sh $PYTHON_VERSION

# ADD aspell /tmp/aspell
ADD docker-data/aspell-cs-0.51-0  /tmp/aspell-cs-0.51-0
ADD aspell-python  /tmp/aspell-python

# install fixed Aspell
WORKDIR /tmp
RUN git clone -b rel-0.60.8.1 https://github.com/GNUAspell/aspell.git
WORKDIR /tmp/aspell
RUN   ./autogen && \
      ./configure && \
      make && \
      make install
RUN ldconfig

# install Aspell dictionary for Czech
WORKDIR /tmp/aspell-cs-0.51-0
RUN   ./configure && \
      make && \
      make install
ENV LANG=cs_CZ

# install Conda
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh && \ 
    bash Miniconda3-latest-Linux-x86_64.sh -b
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN bash /root/miniconda3/etc/profile.d/conda.sh 

SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

RUN conda install -c conda-forge cudatoolkit=11.8.0 && pip install nvidia-cudnn-cu11==8.6.0.163
RUN   CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)")) && \
      export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH && \
      mkdir -p $CONDA_PREFIX/etc/conda/activate.d && \
      echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh && \
      echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# install aspell-python
WORKDIR /tmp
RUN git clone https://github.com/ndvbd/aspell-python.git
WORKDIR  /tmp/aspell-python
RUN   python setup.3.py build &&\
      python setup.3.py install

# prepare code
ADD code /code
WORKDIR /code

RUN python -m pip install -r requirements.txt

ENV PATH="/root/miniconda3/bin:${PATH}"
RUN bash /root/miniconda3/etc/profile.d/conda.sh 
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

RUN git clone https://github.com/ufal/errant_czech.git

RUN cd errant_czech && \
    python -m pip install -e . && \
    python -m pip install -r errant/cs/requirements.txt && \ 
    curl --remote-name-all 'https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1674{/czech-morfflex-pdt-160310.zip}' && \
    unzip czech-morfflex-pdt-160310.zip && \
    cp czech-morfflex-pdt-160310/czech-morfflex-160310.dict errant/cs/resources/