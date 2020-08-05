#!/usr/bin/env bash

if [[ "${OSTYPE}" == 'cygwin' ]]
then
    # set -o igncr # execute it manually in source-conda.sh for now it doesnt work
    # export SHELLOPTS # should be after or before set ?
    source /cygdrive/c/Miniconda3/etc/profile.d/conda.sh
else
  source $(conda info --base)/etc/profile.d/conda.sh
fi