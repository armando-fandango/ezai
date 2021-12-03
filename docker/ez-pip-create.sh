#!/usr/bin/env bash

pip3 install virtualenv

python3 -m venv venv
surce venv/bin/activate
pip3 install --upgrade pip

pip3 install -r docker/requirements-conda.txt
pip3 install -r docker/requirements-pip.txt

