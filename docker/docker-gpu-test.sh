#!/usr/bin/env bash

base="nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04"
docker run --rm --gpus all $base nvidia-smi