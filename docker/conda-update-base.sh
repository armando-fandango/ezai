#!/usr/bin/env bash

conda config --set auto_activate_base false && \
    conda config --env --prepend channels conda-forge && \
    conda config --env --set channel_priority strict && \
    conda config --env --remove channels defaults && \
    conda update -y -n base conda
