import os
import numpy as np

from .models import models
from .util import util
from .util import filesystem_util

from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.configspace import json as smac_json
from smac.scenario.scenario import Scenario

import smac

def configspace_to_json_file(configspace, filename):
    with open(filename, 'w') as f:
        f.write(smac_json.write(configspace))

def configspace_from_json_file(filename):
    with open(filename, 'r') as f:
        return smac_json.read(f.read())

def configsample_from_json_file(filename):
    configspace = configspace_from_json_file(filename)
    return configspace.sample_configuration()

class AutoML_SMAC():
    def __init__(self):
        pass

    def build(self, exp_conf, aml_conf, csfile,
              x_train, y_train, x_valid, y_valid,
              x_test=None, y_test=None):
        self.aml_conf = aml_conf
        self.csfile = csfile
        if aml_conf is not None:
            configspace = configspace_from_json_file(csfile)
            scenario = Scenario({"run_obj": "quality",
                             # we optimize quality (alternatively runtime)
                             "runcount-limit": aml_conf.runcount_limit,
                             "algo_runs_timelimit": aml_conf.algo_runs_timelimit,
                             "cutoff": aml_conf.cutoff,
                             # max. number of function evaluations; for this example set to a low number
                             "cs": configspace,  # configuration space
                             "deterministic": "true",
                             "output_dir": aml_conf.output_dir
                             })
            self.smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
                                 tae_runner=self.smac_tae_runner, run_id=0)
        else:
            self.smac = None
    # save data
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test = x_test
        self.y_test = y_test
        # keep the standard default conf
        self.exp_conf = exp_conf.deepcopy() #.copy() # dont make a copy as we want to update
                                    # original conf so we can save it
        # not sure why you dint want to copy... if it doesnt work remove copy

        self.model = None
        self.metrics = None
        self.t_train = None
        self.best_loss = None

        return self

    def bfpe(self, autotune=True):
        """
        bfpe stands for build, fit, predict, eval

        Notes:
        1. save should be False for autotune and True for non-autotune
        1. needs access to {x,y}_{train,valid,test}
        2. needs access to conf object
        :return:
        """
        save = not autotune

        et = util.ExpTimer()
        # 1 of 5 : Build the Model
        # Needs conf

        # get the class from temporal model
        model = getattr(models, self.exp_conf.mclass)()
        model.build(self.exp_conf)

        # 2 of 5 : Fit the Model
        # Needs train and valid parts, disable the save

        et.start()
        model.fit(self.x_train, self.y_train, self.x_valid, self.y_valid,
                  verbose=0, save=save)
        t_train = et.stop()

        if autotune:
            # 3 of 5 : Predict
            # needs x_test - for now we do it on valid
            y_hat = model.predict(self.x_valid)

            # 4 of 5 : Evaluate
            # needs y_hat, y_test, loss is returned as a list with 1 element
            loss = model.calc_metrics(self.y_valid, y_hat, [self.exp_conf.los_fname])[0]

            # 5 of 5 - bonus save best model
            # Saving will slow down HPT hence not doing it
            return loss
        else:
            self.t_train = t_train
            self.model = model
            if (self.x_test is not None) and (self.y_test is not None):
                self.metrics, self.y_hat = self.model.evaluate(self.x_test, self.y_test)
        return self

    def smac_tae_runner(self, cfg):
        # get the cfg from the ConfigSpace as dictionary
        cfg = {k: cfg[k] for k in cfg if cfg[k]}

        # merge configs
        #conf = self.conf.copy()
        self.exp_conf.update(cfg)

        loss = self.bfpe(autotune=True)
        return float(loss)

    def run(self, autotune=True):
        if self.smac is None:
            cfg = configsample_from_json_file(self.csfile)
            cfg = {k: cfg[k] for k in cfg if cfg[k]}
            self.exp_conf.update(cfg)
        if autotune:
            if self.aml_conf is None:
                print('cant autotune, no aml_conf supplied at build')
            else:
                cfg = self.smac.optimize()
                cfg = {k: cfg[k] for k in cfg if cfg[k]}
                self.exp_conf.update(cfg)

        self.bfpe(autotune=False)
        return self

"""
AUtoML Workflow

Define ConfSpace

Define Scenario Object (confSpace)

Define eval(confSpace)
    Build model with values from confSpace
    fit the model
    predict the model
    evaluate the model

return scores

define smac (scenario, eval func)

best_conf = smac.optimize()
inc_val = eval(best_conf)

"""

"""
This can be also another approach:

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
import sys

X = []
y = []
X_val = []
y_val = []

space = {'choice': hp.choice('num_layers',
                             [ {'layers':'two', },
                               {'layers':'three',
                                'units3': hp.uniform('units3', 64,1024),
                                'dropout3': hp.uniform('dropout3', .25,.75)}
                               ]),

         'units1': hp.uniform('units1', 64,1024),
         'units2': hp.uniform('units2', 64,1024),

         'dropout1': hp.uniform('dropout1', .25,.75),
         'dropout2': hp.uniform('dropout2',  .25,.75),

         'batch_size' : hp.uniform('batch_size', 28,128),

         'nb_epochs' :  100,
         'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
         'activation': 'relu'
         }

def f_nn(params):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import Adadelta, Adam, rmsprop

    print ('Params testing: ', params)
    model = Sequential()
    model.add(Dense(output_dim=params['units1'], input_dim = X.shape[1]))
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout1']))

    model.add(Dense(output_dim=params['units2'], init = "glorot_uniform"))
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout2']))

    if params['choice']['layers']== 'three':
        model.add(Dense(output_dim=params['choice']['units3'], init = "glorot_uniform"))
        model.add(Activation(params['activation']))
        model.add(Dropout(params['choice']['dropout3']))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'])

    model.fit(X, y, nb_epoch=params['nb_epochs'], batch_size=params['batch_size'], verbose = 0)

    pred_auc =model.predict_proba(X_val, batch_size = 128, verbose = 0)
    acc = roc_auc_score(y_val, pred_auc)
    print('AUC:', acc)
    sys.stdout.flush()
    return {'loss': -acc, 'status': STATUS_OK}


trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=50, trials=trials)
print 'best: '
print best
"""

"""
def svm_from_cfg(cfg):
    Creates a SVM based on a configuration and evaluates it on the
    iris-dataset using cross-validation.
    
    Parameters:
    -----------
    cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
    Configuration containing the parameters.
    Configurations are indexable!
    
    Returns:
    --------
    A crossvalidated mean score for the svm on the loaded data-set.
    
    
    # For deactivated parameters, the configuration stores None-values.
    # This is not accepted by the SVM, so we remove them.
    cfg = {k : cfg[k] for k in cfg if cfg[k]}
    # We translate boolean values:
    cfg["shrinking"] = True if cfg["shrinking"] == "true" else False
    # And for gamma, we set it to a fixed value or to "auto" (if used)
    if "gamma" in cfg:
        cfg["gamma"] = cfg["gamma_value"] if cfg["gamma"] == "value" else "auto"
        cfg.pop("gamma_value", None)  # Remove "gamma_value"
    
    clf = svm.SVC(**cfg, random_state=42)
    
    scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    return 1-np.mean(scores)  # Minimize!
    
build space
build eval func
scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                 "runcount-limit": 200,  # maximum function evaluations
                 "cs": cs,               # configuration space
                 "deterministic": "true"
                 })
print("Optimizing! Depending on your machine, this might take a few minutes.")
smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
    tae_runner=svm_from_cfg)

incumbent = smac.optimize()

inc_value = svm_from_cfg(incumbent)

print("Optimized Value: %.2f" % (inc_value))

--------------------------





# Build Configuration Space which defines all parameters and their ranges
configuration_space = ConfigurationSpace()

rate_of_learning = UniformFloatHyperparameter("rate_of_learning", hyperparameter_values_dic['rate_of_learning'][0],
                                              hyperparameter_values_dic['rate_of_learning'][1],
                                              default_value=hyperparameter_values_dic['rate_of_learning'][0])
cell_dimension = UniformIntegerHyperparameter("cell_dimension",
                                              hyperparameter_values_dic['cell_dimension'][0],
                                              hyperparameter_values_dic['cell_dimension'][1],
                                              default_value=hyperparameter_values_dic['cell_dimension'][
                                                  0])
no_hidden_layers = UniformIntegerHyperparameter("num_hidden_layers",
                                                hyperparameter_values_dic['num_hidden_layers'][0],
                                                hyperparameter_values_dic['num_hidden_layers'][1],
                                                default_value=hyperparameter_values_dic['num_hidden_layers'][0])
minibatch_size = UniformIntegerHyperparameter("minibatch_size", hyperparameter_values_dic['minibatch_size'][0],
                                              hyperparameter_values_dic['minibatch_size'][1],
                                              default_value=hyperparameter_values_dic['minibatch_size'][0])
max_epoch_size = UniformIntegerHyperparameter("max_epoch_size", hyperparameter_values_dic['max_epoch_size'][0],
                                              hyperparameter_values_dic['max_epoch_size'][1],
                                              default_value=hyperparameter_values_dic['max_epoch_size'][0])
max_num_of_epochs = UniformIntegerHyperparameter("max_num_epochs", hyperparameter_values_dic['max_num_epochs'][0],
                                                 hyperparameter_values_dic['max_num_epochs'][1],
                                                 default_value=hyperparameter_values_dic['max_num_epochs'][0])
l2_regularization = UniformFloatHyperparameter("l2_regularization",
                                               hyperparameter_values_dic['l2_regularization'][0],
                                               hyperparameter_values_dic['l2_regularization'][1],
                                               default_value=hyperparameter_values_dic['l2_regularization'][0])
gaussian_noise_stdev = UniformFloatHyperparameter("gaussian_noise_stdev",
                                                  hyperparameter_values_dic['gaussian_noise_stdev'][0],
                                                  hyperparameter_values_dic['gaussian_noise_stdev'][1],
                                                  default_value=hyperparameter_values_dic['gaussian_noise_stdev'][
                                                      0])
random_normal_initializer_stdev = UniformFloatHyperparameter("random_normal_initializer_stdev",
                                                             hyperparameter_values_dic[
                                                                 'random_normal_initializer_stdev'][0],
                                                             hyperparameter_values_dic[
                                                                 'random_normal_initializer_stdev'][1],
                                                             default_value=hyperparameter_values_dic[
                                                                 'random_normal_initializer_stdev'][
                                                                 0])

# add the hyperparameter for learning rate only if the  optimization is not cocob
if optimizer == "cocob":
    configuration_space.add_hyperparameters(
        [cell_dimension, no_hidden_layers, minibatch_size, max_epoch_size, max_num_of_epochs,
         l2_regularization, gaussian_noise_stdev, random_normal_initializer_stdev])
else:

    configuration_space.add_hyperparameters(
        [rate_of_learning, cell_dimension, minibatch_size, max_epoch_size,
         max_num_of_epochs, no_hidden_layers,
         l2_regularization, gaussian_noise_stdev, random_normal_initializer_stdev])

# creating the scenario object
scenario = Scenario({
    "run_obj": "quality",
    "runcount-limit": hyperparameter_tuning_configs.SMAC_RUNCOUNT_LIMIT,
    "cs": configuration_space,
    "deterministic": "true",
    "abort_on_first_run_crash": "false"
})

def train_model_smac(configs):
    error, _ = train_model(configs)
    return error
def train_model(configs):
    if "rate_of_learning" in configs.keys():
        rate_of_learning = configs["rate_of_learning"]
        global learning_rate
        learning_rate = rate_of_learning
    cell_dimension = configs["cell_dimension"]
    num_hidden_layers = configs["num_hidden_layers"]
    minibatch_size = configs["minibatch_size"]
    max_epoch_size = configs["max_epoch_size"]
    max_num_epochs = configs["max_num_epochs"]
    l2_regularization = configs["l2_regularization"]
    gaussian_noise_stdev = configs["gaussian_noise_stdev"]
    random_normal_initializer_stdev = configs["random_normal_initializer_stdev"]

    print(configs)

    # select the appropriate type of optimizer
    error, error_list = model_trainer.train_model(num_hidden_layers=num_hidden_layers,
                                      cell_dimension=cell_dimension,
                                      minibatch_size=minibatch_size,
                                      max_epoch_size=max_epoch_size,
                                      max_num_epochs=max_num_epochs,
                                      l2_regularization=l2_regularization,
                                      gaussian_noise_stdev=gaussian_noise_stdev,
                                      random_normal_initializer_stdev=random_normal_initializer_stdev,
                                      optimizer_fn=optimizer_fn)

    print(model_identifier)
    return error, error_list


# optimize using an SMAC object
smac = SMAC(scenario=scenario, rng=np.random.RandomState(seed), tae_runner=train_model_smac)


incumbent = smac.optimize()
return incumbent.get_dictionary()

"""
