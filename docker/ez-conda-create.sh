#!/usr/bin/env bash

#TODO This line doesnt work in source-conda.sh
if [[ "${OSTYPE}" == 'cygwin' ]]
then
  export SHELLOPTS # should be after or before set ?
  set -o igncr # execute it manually for now it doesnt work
  source /cygdrive/c/Miniconda3/etc/profile.d/conda.sh
else
  source $(conda info --base)/etc/profile.d/conda.sh
fi
# add -k if ssl_verify needs to be set to false
opts="-c conda-forge --strict-channel-priority"
pkgs="jupyter notebook jupyter_contrib_nbextensions jupyter_nbextensions_configurator"
if [ -z "$1" ]
then
  venv="/opt/conda/envs/ezai"
else
  venv=$1
fi

conda activate $venv || \
    (echo "${venv} doesnt exist - creating now..." && \
    conda create -y -p $venv $opts python=3.7 $pkgs && \
    conda activate $venv && \
    conda config --env --prepend channels conda-forge && \
    conda config --env --set channel_priority strict && \
    conda config --env --remove channels defaults && \
    conda config --set auto_activate_base false && \
    jupyter nbextension enable code_prettify/code_prettify && \
    jupyter nbextension enable toc2/main)
    #jupyter nbextension enable ipyparallel && \

conda activate $venv && \
    conda install -y -p $venv $opts --file ./requirements-conda.txt #&& \
    #pip install --no-deps --use-feature 2020-resolver -r ./requirements-pip.txt


