#!/usr/bin/env bash

#TODO This line doesnt work in cygwin
if [[ "${OSTYPE}" == 'cygwin' ]]
then
    # set -o igncr
    export SHELLOPTS
else
  source $(conda info --base)/etc/profile.d/conda.sh
fi
# add -k if ssl_verify needs to be set to false
opts="-c conda-forge --override-channels --strict-channel-priority"
pkgs="jupyter notebook jupyter_contrib_nbextensions jupyter_nbextensions_configurator"
if [ -z "$1" ]
then
  venv="./ezai-env"
else
  venv=$1
fi

conda activate $venv || \
    (echo "venv doesnt exist - creating now..." && \
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
    conda install -y -p $venv $opts --file $(dirname $0)/requirements-conda.txt && \
    pip install --no-deps --use-feature 2020-resolver -r $(dirname $0)/requirements-pip.txt


