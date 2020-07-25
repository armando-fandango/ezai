#!/usr/bin/env bash

conda config --env --add channels conda-forge && \
    conda config --env --set channel_priority strict && \
    conda config --set auto_activate_base false && \
    conda update -y -n base conda
