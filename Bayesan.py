import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import keras_tuner
import numpy as np
import Utility as ut
import warnings
from keras import Input, Model

warnings.simplefilter("ignore", UserWarning)
np.random.seed(19740429)
tf.random.set_seed(51)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
from tensorflow.keras.regularizers import L1L2

import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)
import Callbacks as cb
import Models as md

DISPLAY = False

parameters = {
    "debug": False,
    "time_window": 100,
    "min_hidden_layers": 0,
    "max_hidden_layers": 0,
    "future_step": 1,
    "sampling": 1,
    "learning_rate": 7e-6,
    "learning_rate_tg": 6e-7,
    "batch_size": 64,
    "n_epochs": 70,
    'dropout': 0.2,
    "label": 'Pitch',
    "patience": 6,
    "val_split": 0,
    "max_trials": 20,
    "filter_in": 'none',  # kalman wiener simple none
    "filter_out": 'none',  # kalman wiener simple none
    "optimizer": 'adam',  # adam
    "activation": 'tanh',  # tanh or relu
    "scaler": 'Standard',  # Standard or MinMaxScaler
    "loss_function": 'huber_loss'  # huber_loss or mean_squared_error
}
tags = ['LSTM_BO']

ut.set_seed()

X_train, X_test, y_train, y_test = ut.clueanUpData(False, parameters['filter_in'], bestFeature=0)

X_train, X_test = ut.scalingData(X_train, X_test, parameters['scaler'])

X_train, X_test, y_train, y_test = ut.toSplitSequence(X_train, X_test, y_train, y_test, parameters['time_window'],
                                                      parameters['future_step'])

activation = ut.get_activation(parameters['activation'])

# Number history timestamps
n_timestamps = X_train.shape[1]
# Number of input features to the model
n_features = X_train.shape[2]
# Number of output timestamps
n_future = y_train.shape[1]

DEFAULT_RETURN = 0.4

my_callbacks = cb.callbacks(neptune=False,
                                early=parameters['patience'] > 0,
                                lr=True,
                                scheduler=False,
                                run=None,
                                opti=None,
                                target=parameters['learning_rate_tg'],
                                patience=2)
# my_callbacks.append(cb.tensorBoardCallback())

def build_model(hp):
    tf.keras.backend.clear_session()

    # if parameters['max_hidden_layers'] > parameters['min_hidden_layers']:
    #     n_layers = hp.Int('n_layers', parameters['mai_hidden_layers'], parameters['max_hidden_layers'])
    # else:
    #     n_layers = 0

    # l1 = 0.0
    # l2 = 0.0
    l1 = hp.Choice("l1", values=[0.0, 0.01])
    l2 = hp.Choice("l2", values=[0.0, 0.01])

    if parameters['dropout'] > 0:
        recurrent_dropout = hp.Float('recurrent_dropout', min_value=0.0, max_value=parameters['dropout'], step=0.2)
    else:
        recurrent_dropout = 0.0

    input_units = 256 #hp.Int('input_unit', min_value=n_features, max_value=128)
    # output_units = hp.Int('output_units', min_value=256, max_value=768, step=256, default=512)

    # STACKED
    # model = tf.keras.models.Sequential()
    # model.add(LSTM(input_units,
    #                return_sequences=False,
    #                activation=activation,  # tanh
    #                kernel_regularizer=L1L2(l1=l1, l2=l2),
    #                recurrent_dropout=recurrent_dropout,
    #                input_shape=(X_train.shape[1], X_train.shape[2])))
    # # for i in range(n_layers):
    # #     model.add(LSTM(hp.Int(f'lstm_{i}_units', min_value=256, max_value=768, step=512), return_sequences=True))
    # # model.add(LSTM(hp.Int('layer_2_neurons', min_value=128, max_value=384, step=128, default=256)))
    # # # model.add(keras.layers.Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.1,step=0.1)))
    # # # model.add(Dense(hp.Int('dense_neurons', min_value=64, max_value=192, step=64, default= 64), activation='linear'))
    # # # model.add(Dense(units=64, activation='linear'))
    # model.add(Dense(y_train.shape[1], activation='linear'))

    #STATE = TRUE
    # input = Input(shape=(n_timesteps, n_features))
    #
    #
    # lstm1 = LSTM(input_units, return_state=True, activation=activation, kernel_regularizer=L1L2(l1=l1, l2=l2))
    # LSTM_output, state_h, state_c = lstm1(input)
    # states = [state_h, state_c]
    #
    # repeat = RepeatVector(n_timesteps)  # 1
    # LSTM_output = repeat(LSTM_output)
    #
    # lstm2 = LSTM(input_units, return_sequences=True)
    # all_state_h = lstm2(LSTM_output, initial_state=states)
    #
    # # dense = TimeDistributed(Dense(y_train.shape[1], activation='linear'))
    # # output = dense(all_state_h)
    #
    # lstm3 = LSTM(output_units, return_sequences=False)
    # all_state_d = lstm3(all_state_h)
    #
    # dense = Dense(y_train.shape[1], activation='linear')
    # output = dense(all_state_d)
    #
    # model = Model(input, output, name='model_LSTM_return_state')

    # model = tf.keras.models.Sequential()
    # model.add(Bidirectional(LSTM(input_units, activation=parameters['activation'],  kernel_regularizer=L1L2(l1=l1, l2=l2),),
    #                             input_shape=(n_timestamps, n_features)))
    # model.add(Dense(n_future, activation='linear'))


    # -------------------------------- LEARNING RATE -----------------------------

    input = tf.keras.layers.Input(shape=(n_timestamps, n_features))

    lstm1 = LSTM(input_units, return_sequences=True, return_state=True,
                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                 recurrent_dropout=recurrent_dropout,
                 activation=activation)
    all_state_h, state_h, state_c = lstm1(input)
    states = [state_h, state_c]

    lstm2 = LSTM(input_units, return_sequences=False, activation=activation)
    all_state_h = lstm2(all_state_h, initial_state=states)

    dense = Dense(n_future, activation='linear')
    output = dense(all_state_h)

    model = Model(input, output, name='model_LSTM_return_state')

    # lr = 0.001  # default
    # lr = parameters['learning_rate']
    lr = hp.Choice("learning_rate", values=[7e-6, 6e-6, 9e-6])

    opti = ut.get_optimizer(optimizer=parameters['optimizer'],
                            learning_rate=lr)

    model = md.compile_model(model, opti, parameters['loss_function'])

    return model


class MyTuner(keras_tuner.tuners.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 96, step=32, default=64)
        return super(MyTuner, self).run_trial(trial, *args, **kwargs)

tuner = keras_tuner.tuners.BayesianOptimization(#MyTuner(
    build_model,
    objective='val_loss',
    max_trials=parameters['max_trials'],
    executions_per_trial=1,
    overwrite=True,
)

tuner.search(
    x=X_train,
    y=y_train,
    epochs=parameters['n_epochs'],
    batch_size=parameters['batch_size'],
    callbacks=my_callbacks, #reduce_lr  hist_callback
    validation_data=(X_train, y_train)
)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(1)[0]
print(best_hps)

print("START GET MODEL")
best_model = tuner.get_best_models(num_models=1)[0]

best_model.summary()

import PlotResults as pl

pl.PlotResult(best_model, X_test, y_test, None, run=None, batch_size=parameters['batch_size'], n_future=n_future,
              saveDelta=True, SENSOR_ERROR=0.05)

# test_pred = best_model.predict(X_test)
# import sklearn.metrics as metrics
#
# r2 = metrics.r2_score(y_test, test_pred)
# print("R2 Testing: ", r2)
# rmse = np.sqrt(metrics.mean_squared_error(y_test, test_pred))
# print('RMSD ( Root Mean Squared Error ) :', rmse)

print("## DONE ##")
