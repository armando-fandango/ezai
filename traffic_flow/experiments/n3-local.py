#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import time
import sys
from contextlib import redirect_stdout, redirect_stderr

PKG_PATH=Path.home() / 'projects' / 'ezai'
PKG_PATH = str(PKG_PATH.resolve())
if not PKG_PATH in sys.path:
    sys.path.append(PKG_PATH)
print(sys.path)
from traffic_flow.experiments import n3

# 1 : Let's get the arguments
parser = ArgumentParser()
parser.add_argument('-e', '--exp_id')  # experiment id - e.g. e1_1
parser.add_argument('-i', '--exp_iid')  # experiment instance id - e.g. local_1
parser.add_argument('-d', '--exp_did')  # which data -e.g. pems_d5
parser.add_argument('-n', '--n_tid', type=int,
                    default=2)  # task id i.e. which trial
parser.add_argument('-o', '--n_offset_tid', type=int,
                    default=0)  # task id i.e. which trial
parser.add_argument("-t", "--test", help="turn on test mode",
                    action="store_true")
parser.add_argument('-u', '--notune', help="don't auto-tune",
                    action="store_true")
args = parser.parse_args()


exp_id = args.exp_id
exp_iid = args.exp_iid
exp_did = args.exp_did
n_tid = args.n_tid
n_offset_tid = args.n_offset_tid
"""
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
"""

exp_out = Path.home() / 'traffic_flow_exp' / 'out' / exp_id / exp_iid
exp_logs = Path.home() / 'traffic_flow_exp' / 'logs'

# TODO: export JID= get time stamp here
jid = '{}-{}'.format(exp_iid, time.time())  # 1
for tid in range(n_offset_tid + 1, n_offset_tid + n_tid + 1):
    exp_log = exp_logs / "{}-{}.txt".format(jid, tid)
    #exp_code = Path.home() / 'projects' / 'ezai' / 'traffic_flow' / 'experiments' / 'n3.py'
    # EXPRUN_COMMAND="python3  "
    # EXPRUN_COMMAND+=" --exp_id=${EXP_ID} --exp_iid=${EXP_IID} "
    # EXPRUN_COMMAND+=" --exp_tid=${TID} --exp_did=${EXP_DID} "
    # EXPRUN_COMMAND+=" --exp_out=${EXP_OUT} ${EXP_TEST} ${EXP_NOTUNE}"

    print("localrun: running using ${EXPRUN_COMMAND}" > "${EXP_LOG}")
    with open(str(exp_log.resolve()), 'a+') as file:
        with redirect_stdout(file):
            with redirect_stderr(sys.stdout):

                n3.n3(exp_id=exp_id,
                   exp_iid=exp_iid,
                   exp_tid=tid,
                   exp_did=exp_did,
                   exp_out=str(exp_out.resolve()),
                   autotune=not args.notune,
                   testmode=args.test)
#    eval "${EXPRUN_COMMAND}"
# >> ${EXP_LOG} means append the stdout to file
# 2>&1 this means redirect stderr to stdout

# move the log files
    exp_logs1 = exp_out / 'logs'
    exp_logs1.mkdir(parents=True, exist_ok=True)
    exp_log1 = exp_logs1 / '{}.txt'.format(tid)
    exp_log.replace(exp_log1)
    print(exp_log1)
