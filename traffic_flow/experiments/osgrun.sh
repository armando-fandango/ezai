#!/usr/bin/env bash
echo "***** osgrun *****"

export JID=${1}
export TID=${2}

export EXP_ARGS="${HOME}/exp/exp_args/${TID}"
export EXP_OUT="${HOME}/exp/exp_out/${JID}_${TID}"

# transfer data from stashcache
#module load stashcp
#stashcp /user/amitgoel/datasets/caltrans.tar.gz ${HOME}/

# untar data
mkdir ~/datasets
#tar -xzf ${HOME}/caltrans.tar.gz -C ${HOME}/datasets
tar -xzf caltrans.tar.gz -C ~/datasets


# untar python code
tar -xzf dl-ts.tar.gz -C ${HOME}

echo "osgrun: making argument folder ${EXP_ARGS}"
# copy the exp_args file
mkdir -p ${EXP_ARGS}
echo "osgrun: Copying argument file"
mv ${TID}_args.csv  ${EXP_ARGS}

echo "osgrun: making output folder ${EXP_OUT}"
mkdir -p ${EXP_OUT}


export EXP_LOG="${EXP_OUT}/${TID}_log.txt"

export EXPRUN_COMMAND="~/ezml/scripts/exprun.sh ${JID} ${TID} ${EXP_ARGS} ${EXP_OUT}"

echo "osgrun: running exprun.sh"
echo "osgrun: ${EXPRUN_COMMAND}"
# run the code
eval "${EXPRUN_COMMAND}"

tarzip stuff again
