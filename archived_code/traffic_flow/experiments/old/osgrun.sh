#!/usr/bin/env bash

set -e

echo "***** osgrun *****"

export JID=${1}
export TID=${2}
export EXP_ID=${3}
export EXP_IID=${4}
export EXP_DID=${5}

export EXP_OUT="${HOME}/traffic_flow_exp/out/${EXP_ID}/${EXP_IID}"
export EXP_LOGS="${HOME}/traffic_flow_exp/logs"

#export EXP_ARGS="${HOME}/exp/exp_args/${TID}"
#export EXP_OUT="${HOME}/exp/exp_out/${JID}_${TID}"

mkdir -p ${EXP_OUT}
mkdir -p ${EXP_LOGS}

# transfer data from stashcache
#module load stashcp
#stashcp /user/amitgoel/datasets/caltrans.tar.gz ${HOME}/

# untar data
mkdir -p ~/traffic_flow_exp/data
mkdir -p ~/projects
#tar -xzf ${HOME}/caltrans.tar.gz -C ${HOME}/datasets
tar -xzf data.tar.gz -C ~/

# untar python code
tar -xzf ezai.tar.gz -C ~/

export EXP_LOG="${EXP_LOGS}/${JID}-${TID}.txt"

#export EXP_LOG="${EXP_OUT}/${TID}_log.txt"

#export EXPRUN_COMMAND="~/ezml/scripts/exprun.sh ${JID} ${TID} ${EXP_ARGS} ${EXP_OUT}"
export EXPRUN_COMMAND="python3 ${HOME}/projects/ezai/traffic_flow/experiments/n3.py "
EXPRUN_COMMAND+=" --exp_id=${EXP_ID} --exp_iid=${EXP_IID} "
EXPRUN_COMMAND+=" --exp_tid=${TID} --exp_did=${EXP_DID} "
EXPRUN_COMMAND+=" --exp_out=${EXP_OUT}"

echo "osgrun: running exprun.sh"
echo "osgrun: ${EXPRUN_COMMAND}"
# run the code
eval "${EXPRUN_COMMAND}"

#echo "localrun: running using ${EXPRUN_COMMAND}" > "${EXP_LOG}"
#  sync

#  eval "${EXPRUN_COMMAND}" >> ${EXP_LOG} 2>&1

  # move the log files
#  export EXP_LOGS1="${EXP_OUT}/logs"
#  mkdir -p ${EXP_LOGS1}
#  export EXP_LOG1="${EXP_LOGS1}/${TID}.txt"
#  mv ${EXP_LOG} ${EXP_LOG1}

#  echo ${EXP_LOG1}
# done

