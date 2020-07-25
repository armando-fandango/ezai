#!/usr/bin/env bash

source $(conda info --base)/etc/profile.d/conda.sh

conda env create -f ./conda-ez-create.yaml && \
    conda init bash && \
    conda activate ezai && \
    conda config --env --add channels conda-forge && \
    conda config --env --set channel_priority strict && \
    conda config --set auto_activate_base false && \
    jupyter nbextension enable code_prettify/code_prettify && \
    jupyter nbextension enable toc2/main
    #jupyter nbextension enable ipyparallel && \

#. /opt/conda/etc/profile.d/conda-ez-update.sh

