#!/usr/bin/env bash

# TODO: modify EXPID, EXPDID below

#[[ -z "${DEPLOY_ENV}" ]] && MyVar='default' || MyVar="${DEPLOY_ENV}"

# We might have data set, its subset and subset....
export EXP_ID='n3_1'
export EXP_IID='local_test'
export EXP_DID='samiul_i75'
export N_TID=2 #576
export N_OFFSET_TID=0

while getopts ":e:i:d:n:o:tu" opt; do
  case $opt in
    e) EXP_ID="$OPTARG"
    ;;
    i) EXP_IID="$OPTARG"
    ;;
    d) EXP_DID="$OPTARG"
    ;;
    n) N_TID="$OPTARG"
    ;;
    o) N_OFFSET_TID="$OPTARG"
    ;;
    t) EXP_TEST="-t"
    ;;
    u) EXP_NOTUNE="-u"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

export EXP_OUT="${HOME}/traffic_flow_exp/out/${EXP_ID}/${EXP_IID}"
export EXP_LOGS="${HOME}/traffic_flow_exp/logs"
mkdir -p ${EXP_OUT}
mkdir -p ${EXP_LOGS}

#TODO: export JID= get time stamp here
export JID=$(date +"%Y-%m-%d_%H-%M-%S")  #1
for TID in $(seq $(( $N_OFFSET_TID + 1 )) $(( $N_OFFSET_TID + $N_TID )))
#{1..$N_TID}
do

  export EXP_LOG="${EXP_LOGS}/${JID}-${TID}.txt"

  export EXPRUN_COMMAND="python3 ${HOME}/projects/ezai/traffic_flow/experiments/n3.py "
  EXPRUN_COMMAND+=" --exp_id=${EXP_ID} --exp_iid=${EXP_IID} "
  EXPRUN_COMMAND+=" --exp_tid=${TID} --exp_did=${EXP_DID} "
  EXPRUN_COMMAND+=" --exp_out=${EXP_OUT} ${EXP_TEST} ${EXP_NOTUNE}"

  echo "localrun: running using ${EXPRUN_COMMAND}" > "${EXP_LOG}"
  sync

  eval "${EXPRUN_COMMAND}" >> ${EXP_LOG} 2>&1

  # move the log files
  export EXP_LOGS1="${EXP_OUT}/logs"
  mkdir -p ${EXP_LOGS1}
  export EXP_LOG1="${EXP_LOGS1}/${TID}.txt"
  mv ${EXP_LOG} ${EXP_LOG1}

  echo ${EXP_LOG1}
done

