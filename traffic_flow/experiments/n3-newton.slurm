#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4  # no more than 16/GPU
#SBATCH --mem-per-cpu=5900
#SBATCH --job-name=strat-exp
#SBATCH --time=24:00:00
#SBATCH --account=pwiegand
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu16
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=armando@ucf.edu
# ### SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --signal=SIGINT@600
#SBATCH -a 1-2
#SBATCH --error=expout/logs/%A-%a.txt
#SBATCH --output=expout/logs/%A-%a.txt
#SBATCH --partition=preemptable
#SBATCH --qos=preemptable
#SBATCH --requeue

### SLURM Specific

export JID=${SLURM_ARRAY_JOB_ID}

export TID=${SLURM_ARRAY_TASK_ID}
#export NEW_TID=`expr $TID + 1000`  # for jobs above 1000

### Common

export EXPID=$1
export EXPDID=$2

export EXPOUT="${HOME}/phd/expout"
export EXPLOGS="${EXPOUT}/logs"  #make it manually in newton else it wont work
#mkdir -p ${EXPOUT}
mkdir -p ${EXPLOGS}

### inside loop common

export EXPLOG="${EXPLOGS}/${JID}-${TID}.txt"
export EXPRUN_COMMAND="python3 ${HOME}/phd/ezml/experiments/n11.py --expid=${EXPID} --exptid=${TID} --expdid=${EXPDID} --expout=${EXPOUT}"
#export EXPRUN_COMMAND="python3 ${HOME}/phd/ezml/experiments/n11.py --expid=${EXPID} --exptid=${NEW_TID} --expout=${EXPOUT}" # for jobs above 1000


## inside loop SLURM Specific
module load cuda/cuda-10.1 anaconda/anaconda3
source activate phd-gpu

echo "newton: running exprun.py using ${EXPRUN_COMMAND}"
sync
cd $SLURM_SUBMIT_DIR
echo "JobID: $JID"
echo "Running on $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."

## inside loop common


eval "${EXPRUN_COMMAND}"

sleep 3

# move the log files
export EXPLOGS1="${EXPOUT}/${EXPID}/logs"
mkdir -p ${EXPLOGS1}
export EXPLOG1="${EXPLOGS1}/${TID}.txt"
#export EXPLOG1="${EXPLOGS1}/${NEW_TID}.txt" # for jobs above 1000

mv ${EXPLOG} ${EXPLOG1}
