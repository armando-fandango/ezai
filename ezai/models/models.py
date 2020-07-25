from tensorflow.keras import layers as k_layers
from tensorflow.keras import models as k_models
from tensorflow import keras
import tensorflow as tf
from os import path as os_path
from pickle import dump as pickle_dump
from tensorflow.keras.callbacks import EarlyStopping as k_EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint as k_ModelCheckpoint
import torch
from ..util import pt_util
from ..util import dict_util

from .. import metrics

class TemporalModel(object):
    def __init__(self):
        pass

    def build(self, conf:dict_util.DictObj):
        # TODO: the following doesnt work out of blue sometimes
        # remove it when you can
        # if not isinstance(conf, Conf):
        #    conf = Conf(conf)

        # use the same vocabulary as in utils.tdutil.td_to_xy

        # [n_tx i.e timesteps_in, n_ty i.e. timesteps_out]
        self.n_tx = conf.n_tx
        self.n_ty = conf.n_ty

        self.n_vx = conf.n_vx  # total inputs n_cx = n_tx * n_vx
        self.n_vy = conf.n_vx  # total outputs = n_cy = n_ty * n_vy

        self.n_cx = self.n_tx * self.n_vx
        self.n_cy = self.n_ty * self.n_vy

        self.layers = conf.get('layers')
            #if hasattr(conf, 'layers') else None
        # [n,n,n,n,n] # Number of LSTM blocks in each layer,
        # first number is first hidden layer,
        # output layer not included here
        # visible input layer is of dimension (n_batch,n_timesteps, n_vx)

        self.n_layers = conf.get('n_layers',
                                 1 if self.layers is None else len(self.layers) )
        self.n_units = conf.get('n_units')
        self.n_epochs = conf.n_epochs
        self.n_batch_size = conf.n_batch_size  # number of rows in a batch

        self.lay_fname = conf.lay_fname  # LSTM, GRU, SimpleRNN
        self.lay_f = getattr(k_layers, self.lay_fname)
        self.los_fname = conf.los_fname
        self.los_f = metrics.getfunc('{0}_k'.format(self.los_fname))
        self.opt_fname = conf.opt_fname

        self.met_fnames = conf.met_fnames
        self.bidir = conf.bidir or False

        self.modelpath = conf.modelpath or 'TemporalModel'

        self.act_fname = conf.act_fname
        # self.act_f = getattr(k_layers, self.act_fname)

        # self.dropout = conf['dropout']
        self.trained = False
        self.loaded_model = False
        return self

    def evaluate(self, x, y):
        y_hat = self.predict(x)
        results = self.calc_metrics(y, y_hat)
        return results, y_hat

    def calc_metrics(self, y, y_hat, metrics_names=None):
        if metrics_names is None:
            metrics_names = [m for m in [self.los_fname] + self.met_fnames]
        metrics_fn = [metrics.getfunc('{}_np'.format(m)) for m in metrics_names]
        results = [fn(y, y_hat) for fn in metrics_fn]
        return results


class RNN_Keras(TemporalModel):
    def __init__(self):
        TemporalModel.__init__(self)

    def build(self, conf, load_saved=False):
        TemporalModel.build(self, conf)
        self.modelpath = conf.modelpath or 'RNN_Keras'

        layer_in = keras.Input(shape=(self.n_tx, self.n_vx), name='layer_in')

        for i in range(0, self.n_layers):
            ret_seq = True if i < self.n_layers - 1 else False
            units = self.layers[i] if (hasattr(self,
                                               'layers') and self.layers) else self.n_units
            rnn_layer = self.lay_f(
                units=units,
                activation=self.act_fname,
                return_sequences=ret_seq,
                name='layer_{}'.format(i + 1))
            if self.bidir:
                rnn_layer = k_layers.Bidirectional(rnn_layer)
            if i == 0:
                layer = rnn_layer(layer_in)
            else:
                layer = rnn_layer(layer)

        # because n_vy is fixed to 1 hence we are using n_ty
        layer_out = k_layers.Dense(units=self.n_ty, name='layer_out')(
            layer)  # no. of outputs

        model = keras.Model(inputs=layer_in, outputs=layer_out)

        met_f = [metrics.getfunc('{}_k'.format(m)) for m in
                 self.met_fnames]
        model.compile(loss=self.los_f, optimizer=self.opt_fname,
                      metrics=met_f)
        self.model = model

        return self

    def fit(self, x_train, y_train, x_valid, y_valid, verbose=0, save=True):
        # make train and valid tensorflow Datasets
        train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(
            self.n_batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(
            self.n_batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)

        model = self.model

        es = k_EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                             patience=min(50, self.n_epochs * 0.20),
                             restore_best_weights=True)
        callbacks = [es]
        if save:
            mc = k_ModelCheckpoint(self.modelpath + '-model.h5',
                                   monitor='val_loss', mode='min',
                                   save_best_only=True)
            callbacks.append(mc)

        self.history = model.fit(x=train,
                                 validation_data=valid,
                                 epochs=self.n_epochs,
                                 verbose=verbose, callbacks=callbacks)

        if save:
            self.save_history()

        self.trained = True
        return self

    def predict(self, x):
        return self.model.predict(x, batch_size=self.n_batch_size)

    # should not be implemented since we inherit from Temporal Model
    # assumes fit has already been done before
    # def evaluate(self, x, y):
    #    return self.model.evaluate(x, y, batch_size=self.n_batch_size,
    #                               verbose=0)

    def save_model(self, modelpath=None):
        modelpath = modelpath or self.modelpath
        if self.model:
            # self.model.save(modelpath + '-savemodel', save_format='tf')
            self.model.save(modelpath + '-model.h5')
        return self

    def save_history(self, modelpath=None):
        modelpath = modelpath or self.modelpath
        if self.history:
            pickle_dump(self.history.history,
                        open(modelpath + '-history.pkl', 'wb'))
            pickle_dump(self.history.params,
                        open(modelpath + '-params.pkl', 'wb'))
        return self

    # TODO should we remove it or keep it ? Keep it as autotune uses it.
    def save(self, modelpath=None):
        self.save_model(modelpath)
        self.save_history(modelpath)
        return self

    def load(self, filename):
        if os_path.exists(filename):
            self.model = k_models.load_model(filename, custom_objects={
                'smape_k': metrics.smape_k})
        else:
            raise (ValueError('No model found at ', filename))
        return self

    def summary(self):
        return self.model.summary()


class LSTM_Keras(RNN_Keras):
    def __init__(self):
        RNN_Keras.__init__(self)

    def build(self, conf, load_saved=False):
        conf.lay_fname = 'LSTM'
        RNN_Keras.build(self, conf, load_saved)


class GRU_Keras(RNN_Keras):
    def __init__(self):
        RNN_Keras.__init__(self)

    def build(self, conf, load_saved=False):
        conf.lay_fname = 'GRU'
        RNN_Keras.build(self, conf, load_saved)


class BiLSTM_Keras(LSTM_Keras):
    def __init__(self):
        LSTM_Keras.__init__(self)

    def build(self, conf, load_saved=False):
        conf.bidir = True
        LSTM_Keras.build(self, conf, load_saved)


class BiGRU_Keras(GRU_Keras):
    def __init__(self):
        GRU_Keras.__init__(self)

    def build(self, conf, load_saved=False):
        conf.bidir = True
        GRU_Keras.build(self, conf, load_saved)


"""
PyTorch

LSTM
defined as :  input_size, hidden_size, num_layers 
                n_vx, hidden layer cells (< n_ty means autoencoder), # of layers (>1 means stacked)

input: input (seq_len, batch, input_size), h_0 ( num_layers * {1 or 2} , batch, hidden_size), c_0 (same as h_0) h_0, c_0 default to zero
                n_tx, batch, n_vx
                or batch, n_tx, n_vx if batch_first is True

output: output (seq_len, batch, {1 or 2 } * hidden_size),  h_n , c_n

LSTMCell

define as: input_size, hidden_size
input: input, h_0, c_0
output: h_1, c_1
"""


class RNN_Torch():
    def __init__(self, conf):
        self.conf = conf
        self.model = Sequence(self.conf)
        self.los_f = torch.nn.MSELoss()
        self.opt_f = torch.optim.Adam(self.model.parameters())

    def fit(self, x_train, y_train, x_valid, y_valid, verbose=0, save=True):
        # make training and test sets in torch
        # x_valid = torch.from_numpy(x_valid).type(torch.Tensor)
        # y_valid = torch.from_numpy(y_valid).type(torch.Tensor).view(-1)
        train = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(x_train).type(torch.Tensor).view(
                    [-1, self.conf.n_tx, self.conf.n_vx]),
                torch.from_numpy(y_train).type(torch.Tensor).view(
                    [-1, self.conf.n_ty])),
            batch_size=self.conf.n_batch_size, shuffle=False)
        #        x_train = x_train.
        # x_valid = X_test.view([self.conf.n_vx, -1, 1])
        print('x_train and y_train after making tensor:', x_train.shape,
              y_train.shape)

        for epoch in range(self.conf.n_epochs):
            # model.hidden = model.init_hidden() # remove this for LSTM to be stateful
            for x_batch, y_batch in train:

                y_pred = self.model(x_batch)  # forward pass
                loss = self.los_f(y_pred, y_batch)
                self.opt_f.zero_grad()  # set gradients to zero
                loss.backward(retain_graph=True)  # upgrade gradients
                self.opt_f.step()  # update params
        return self

    def predict(self, x_test):
        with torch.no_grad():
            y_pred = self.model(x_test)
            y_pred = y_pred.detach().numpy()
        return y_pred

    def summary(self):
        ptutil.summary(self.model, (
            self.conf.n_tx, self.conf.n_batch_size, self.conf.n_vx),
                       device='cpu')


class Sequence(torch.nn.Module):
    def __init__(self, conf):
        torch.nn.Module.__init__(self)
        self.conf = conf
        # self.input_dim = input_dim
        # self.hidden_dim = hidden_dim
        # self.batch_size = batch_size
        # self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = torch.nn.LSTM(self.conf.n_vx, self.conf.n_units,
                                  self.conf.n_layers, batch_first=True)

        # Define the output layer
        self.layer_out = torch.nn.Linear(self.conf.n_units, self.conf.n_ty)

        self.init_hc_to_zero()

    def init_hc_to_zero(self):
        self.hc = (torch.zeros(self.conf.n_layers, self.conf.n_batch_size,
                               self.conf.n_units),
                   torch.zeros(self.conf.n_layers, self.conf.n_batch_size,
                               self.conf.n_units))

    def forward(self, input):
        print('input_shape=', input.shape)
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hc = self.lstm(input, self.hc)
        print('lstm_out = ', lstm_out.shape)
        # Only take the output from the final timestep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        # y_pred = self.layer_out(lstm_out[:,-self.conf.n_ty,:])
        y_pred = self.layer_out(lstm_out[:, -1, :])  # take the last time step
        print('y_pred =', y_pred.shape)
        return y_pred


# TODO Work in progress not finished yet

class AutoArima(TemporalModel):
    def __init__(self):
        TemporalModel.__init__(self)

    def build(self, conf, load_saved=False):
        TemporalModel.build(self, conf)
        self.model = model

    def fit(self, x_train, y_train, x_valid, y_valid, verbose=0, save=True):
        model = self.model

    def predict(self, x):
        # in arima you fit and then predict at same time and there is no separate fit
        from pmdarima.arima import auto_arima
        model = auto_arima(x, trace=True, error_action='ignore',
                           suppress_warnings=True)
        model.fit(x)
        forecast = model.predict(n_periods=self.n_ty)
        return forecast
