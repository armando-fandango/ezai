# Usage:

# Python file to run the experiment after reading values from command line
from argparse import ArgumentParser
import os
import sys
import csv

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2', '3'}
import tensorflow as tf

#EXP_ROOT = os.path.join(os.path.expanduser('~'), 'phd')

# load ezai
#TODO: Take this setting to top so we can change at will
EZAI_ROOT=os.path.join(os.path.expanduser('~'),'projects','ezai')
if not EZAI_ROOT in sys.path:
    sys.path.append(EZAI_ROOT)
import ezai

from ezai.util import util
util.m_info([ezai])

from ezai.util import util, filesystem_util, dict_util
from ezai.util.filesystem_util import makedir
from ezai.util.dict_util import load_dict_from_json
from ezai import automl


# 1 : Let's get the arguments
parser = ArgumentParser()
parser.add_argument('--exp_id')  # experiment id - e.g. e1_1
parser.add_argument('--exp_iid')  # experiment instance id - e.g. local_1
parser.add_argument('--exp_tid', type=int)  # task id i.e. which trial
parser.add_argument('--exp_did') # which data -e.g. pems_d5
parser.add_argument('--exp_out')  # out folder      id / out / iid
parser.add_argument("-t", "--test", help="turn on test mode",
                    action="store_true")
parser.add_argument('-u', '--notune', help="don't auto-tune",
                    action="store_true")
args = parser.parse_args()

if args.test:
    print("Running in test mode... with autotune = {}".format(not args.notune))

exp_id = args.exp_id
exp_iid = args.exp_iid
exp_tid = args.exp_tid
exp_did = args.exp_did
exp_out = args.exp_out

script_folder, _ = os.path.split(sys.argv[0]) #where is this program
exp_conf = dict_util.DictObj(os.path.join(script_folder,
        '{}-{}-exp_conf.json'.format(exp_id, exp_did)))

trial_lists = dict_util.DictObj(os.path.join(script_folder,
        '{}-{}-trial_lists.json'.format(exp_id, exp_did)))
print('trial lists:',trial_lists.dumps_json())

autotune = not args.notune

makedir(exp_out)

#expdid = '{}-{}-{}'.format(args.expdid, exp_conf.n_vx, exp_conf.n_agg)
#data_root = os.path.join(os.path.expanduser('~'), 'traffic_flow_exp', exp_id, 'data')

trial = exp_tid

def get_trial(n, list):
    item = list[n % len(list)]
    n = n // len(list)
    return n,item

trial, exp_conf.road = get_trial(trial, trial_lists.road)
trial, exp_conf.direction = get_trial(trial, trial_lists.direction)
trial, exp_conf.n_vx = get_trial(trial, trial_lists.n_vx)
trial, exp_conf.n_agg = get_trial(trial, trial_lists.n_agg)

# one more trial list to go through below
data_folder_str = '{}-{}-{}-{}-{}'.format(exp_conf.source_data,
                                   exp_conf.road,
                                   exp_conf.direction,
                                   exp_conf.n_vx,
                                   exp_conf.n_agg)

data_folder = os.path.join(os.path.expanduser('~'),
                           'traffic_flow_exp','data',exp_id,data_folder_str)

id_list = load_dict_from_json(os.path.join(data_folder,'id_list.json'))['id_list']
trial, exp_conf.id = get_trial(trial, id_list)

trial, exp_conf.derived_features = get_trial(trial, trial_lists.derived_features)
data_folder = os.path.join(data_folder, '{}'.format(exp_conf.derived_features))

trial, exp_conf.mclass = get_trial(trial, trial_lists.mclass)

#TODO: Should be in smac.. why is this here ?
trial, exp_conf.n_layers = get_trial(trial, trial_lists.n_layers)


# now let us create all folders

logs_folder = os.path.join(exp_out, 'logs')
makedir(logs_folder)

out_folder = exp_out

models_folder = os.path.join(out_folder, 'models')
makedir(models_folder)
modelpath = os.path.join(models_folder, '{}'.format(exp_tid))
exp_conf.modelpath = modelpath

preds_folder = os.path.join(out_folder, 'preds')
makedir(preds_folder)

confs_folder = os.path.join(out_folder, 'confs')
makedir(confs_folder)

metrics_folder = os.path.join(out_folder, 'metrics')
makedir(metrics_folder)

# create results file
metrics_filename = os.path.join(metrics_folder, '{}.csv'.format(exp_tid))
with open(metrics_filename, mode='w') as metrics_file:
    writer = csv.writer(metrics_file,
                        delimiter=',',
                        quotechar='"',
                        quoting=csv.QUOTE_MINIMAL)
    writer.writerow(
        ['exptid', 'id', 't_train', 't_autotune',
         exp_conf.los_fname] + exp_conf.met_fnames)
    metrics_file.flush()

if autotune:
    aml_conf = dict_util.DictObj({
        "run_obj": "quality", # we optimize quality (alternatively runtime)
        "runcount_limit": 3 if args.test else 20,
        "algo_runs_timelimit": 4 * 60 * 60,
        "cutoff": 1 * 60 * 60,
        # max. number of function evaluations; for this example set to a low number
        "deterministic": "true",
        "output_dir": os.path.join(out_folder, 'autotune', '{}'.format(exp_tid))
    })
    filesystem_util.makedir(aml_conf.output_dir)
else:
    aml_conf = None
csfile = os.path.join(script_folder, '{}-configspace.json'.format(exp_id))
# read data
xy = np.load(os.path.join(data_folder, '{}.npz'.format(exp_conf.id)))

print('The shape of x and y from npz file:',xy['x'].shape,xy['y'].shape)

# TODO: why 14400 ?
n_rows = 100 if args.test else 14400
n_rows = min(xy['x'].shape[0] , n_rows)

print('taking ',n_rows,' rows')
x_train, x_valid, x_test = util.tvt_split(xy['x'][-n_rows:] )
y_train, y_valid, y_test = util.tvt_split(xy['y'][-n_rows:] )

print('shapes of x train, valid and test :',x_train.shape,x_valid.shape,x_test.shape)
print('shapes of x train, valid and test :',y_train.shape,y_valid.shape,y_test.shape)
# read config associated with data
data_conf = dict_util.DictObj(os.path.join(data_folder, '{}-conf.json'.format(exp_conf.id)))
exp_conf.update(data_conf)

logstr = '{} {} '.format(exp_conf.id, exp_conf.mclass)
print(logstr)

aml = automl.AutoML_SMAC()
print('Before AutoML:',exp_conf.dumps_json())
aml.build(exp_conf, aml_conf, csfile, x_train, y_train, x_valid, y_valid,
          x_test, y_test)
print('After AutoML:',exp_conf.dumps_json())
print(logstr, ' Training model {} with {} autotune={}...'.format(exp_tid, exp_conf.mclass,
                                                                 autotune))
# save the conf file now, we shall save again after aml
confs_file = os.path.join(confs_folder, '{}.json'.format(exp_tid))
aml.exp_conf.save_to_json(confs_file)

sys.stdout.flush()

et = util.ExpTimer()
et.start()
aml.run(autotune=autotune)
t_autotune = et.stop()
print(logstr, ' Training model ... done')

# save the conf file now that aml has updated
aml.exp_conf.save_to_json(confs_file)

# save predictions
preds_file = os.path.join(preds_folder, '{}'.format(exp_tid))
np.savez_compressed(preds_file, x_test=x_test, y_test=y_test, y_hat=aml.y_hat)
# print(logstr,t_train,t_eval,results)

# save results
with open(metrics_filename, mode='a+') as metrics_file:
    writer = csv.writer(metrics_file,
                        delimiter=',',
                        quotechar='"',
                        quoting=csv.QUOTE_MINIMAL)
    writer.writerow(
        [exp_tid, exp_conf.id, aml.t_train, t_autotune] + aml.metrics)
    metrics_file.flush()

sys.stdout.flush()
