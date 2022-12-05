import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional
import keras_tuner
from keras_tuner import BayesianOptimization
import numpy as np
import pandas as pd
from numpy import hstack, minimum
from pandas import DataFrame, concat
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from numpy import mean, concatenate
from numpy import array, hstack
from pickle import dump, load
import Utility as ut
import warnings
from keras import Input, Model
from keras.layers import PReLU, RepeatVector, TimeDistributed
from tensorflow.keras.activations import elu, relu, tanh

warnings.simplefilter("ignore", UserWarning)
np.random.seed(19740429)
tf.random.set_seed(51)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
from tensorflow.keras.regularizers import L1L2

import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)
from scipy import signal

DISPLAY = False

parameters = {
    "debug": False,
    "time_window": 40,
    "min_hidden_layers": 0,
    "max_hidden_layers": 0,
    "future_step": 1,
    "sampling": 1,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "n_epochs": 70,
    'dropout': 0,
    "label": 'Pitch',
    "patience": 6,
    "val_split": 0,
    "max_trials": 20,
    "optimizer": 'adam',  # adam
    "activation": 'tanh',  # tanh or relu
    "scaler": 'Standard',  # Standard or MinMaxScaler
    "loss_function": 'huber_loss'  # huber_loss or mean_squared_error
}
tags = ['LSTM_BO']

import DataRetrive as es

data = es.retriveDataSet(False)

if (data is None):
    data = es.getDataSet()
    es.saveDataSet(data)
if (data.empty):
    print('Is the DataFrame empty!')
    raise SystemExit

# ['Datetime','SinkMin_AP','YawRate','Bs','Heel','Pitch', 'Lwy', 'Tws']
data = data.drop(['Datetime', 'Bs', 'YawRate', 'Heel', 'Lwy', 'Tws', 'SinkMin_AP'], axis=1)
data.dropna(inplace=True)
# data.info()


if parameters['sampling'] > 1:
    data = data.rolling(parameters['sampling']).mean()
    data = data.iloc[::parameters['sampling'], :]
    data.dropna(inplace=True)

# corr_matrix = data.corr()

# get the label array
yi = data[parameters['label']].values
# get the days of sailing
zi = data['Day'].values
# get the index of the last racing day to be used as TestSet
itemindex = np.where(zi == 6)
test_index = itemindex[0][0]
test_index += 10000

# remove from the DataFrame the colum of the label
data = data.drop(parameters['label'], axis=1)
Xi = data.values
yi = yi.reshape((len(yi), 1))

# data.info()

TEST_SPLIT = 1 - (test_index / len(Xi))  # Test Split to last racing day

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Xi, yi, test_size=TEST_SPLIT, shuffle=False)

# Split Validation Set
if parameters['val_split'] == 0:
    X_val = X_test
    y_val = y_test
else:
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=parameters['val_split'],
                                                      shuffle=False)  # 0.25 x 0.8 = 0.2

# Scale the 3 dataset
if parameters['scaler'] == 'Standard':
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
else:
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(-1, 1))

X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)
X_test = scaler.fit_transform(X_test)

# covert into input/output
dataset = np.append(X_train, y_train, axis=1)
X_train, y_train = ut.split_sequences(dataset, parameters['time_window'], parameters['future_step'])
dataset = np.append(X_val, y_val, axis=1)
X_val, y_val = ut.split_sequences(dataset, parameters['time_window'], parameters['future_step'])
dataset = np.append(X_test, y_test, axis=1)
X_test, y_test = ut.split_sequences(dataset, parameters['time_window'], parameters['future_step'])

# FreeUp some memeory
del (dataset)
del (data)
del (Xi)
del (yi)
del (zi)

n_features = X_train.shape[2]
n_label = y_train.shape[1]


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=parameters['patience'],
                                                  mode='min',
                                                  restore_best_weights=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.01,
                                                 patience=2,
                                                 verbose=1,
                                                 mode='min',
                                                 # min_delta=0.0001,
                                                 cooldown=0,
                                                 min_lr=1e-5)

# run parameter
from datetime import datetime
log_dir = "logs/" + datetime.now().strftime("%m%d-%H%M")

hist_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    embeddings_freq=1,
    write_graph=True,
    write_images= True,
    update_freq='epoch')

print("log_dir", log_dir)

if parameters['activation'] == "relu":
    activation = relu
elif parameters['activation'] == "elu":
    activation = elu
else:
    activation = tanh

n_timestamps = X_train.shape[1]
n_features = X_train.shape[2]
n_future = y_train.shape[1]


def build_model(hp):
    tf.keras.backend.clear_session()

    # if parameters['max_hidden_layers'] > parameters['min_hidden_layers']:
    #     n_layers = hp.Int('n_layers', parameters['mai_hidden_layers'], parameters['max_hidden_layers'])
    # else:
    #     n_layers = 0

    if parameters['optimizer'] == 'adam':
        lr = 0.001 #default
        # lr = parameters['learning_rate']
        # lr = hp.Choice("learning_rate", values=[5e-6, 1e-4, 1e-5])
        opti = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        # lr = 0.01 #default
        # lr = parameters['learning_rate']
        lr = hp.Choice("learning_rate", values=[0.01, 0.001])
        momentum = hp.Choice("momentum", values=[0.9, 0.8, 0.7]) # 0.9 default
        nesterov = True  # hp.Choice("nesterov", values=[True, False])
        opti = tf.keras.optimizers.SGD(learning_rate=lr, decay=1e-6, momentum=momentum, nesterov=nesterov)

    l1 = 0.0
    l2 = 0.0
    # l1 = hp.Choice("l1", values=[0.0, 0.01])
    # l2 = hp.Choice("l2", values=[0.0, 0.01])

    if parameters['dropout'] > 0:
        recurrent_dropout = hp.Float('recurrent_dropout', min_value=0.0, max_value=parameters['dropout'], step=0.1)
    else:
        recurrent_dropout = 0.0

    input_units = hp.Int('input_unit', min_value=n_features, max_value=128)
    # output_units = hp.Int('output_units', min_value=256, max_value=768, step=256, default=512)

    # STACKED
    model = tf.keras.models.Sequential()
    model.add(LSTM(input_units,
                   return_sequences=False,
                   activation=parameters['activation'],  # tanh
                   kernel_regularizer=L1L2(l1=l1, l2=l2),
                   recurrent_dropout=recurrent_dropout,
                   input_shape=(X_train.shape[1], X_train.shape[2])))
    # for i in range(n_layers):
    #     model.add(LSTM(hp.Int(f'lstm_{i}_units', min_value=256, max_value=768, step=512), return_sequences=True))
    # model.add(LSTM(hp.Int('layer_2_neurons', min_value=128, max_value=384, step=128, default=256)))
    # # model.add(keras.layers.Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.1,step=0.1)))
    # # model.add(Dense(hp.Int('dense_neurons', min_value=64, max_value=192, step=64, default= 64), activation='linear'))
    # # model.add(Dense(units=64, activation='linear'))
    model.add(Dense(y_train.shape[1], activation='linear'))

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

    model.compile(optimizer=opti,
                  loss=parameters['loss_function'],
                  metrics=[tf.keras.metrics.RootMeanSquaredError(),
                           tf.keras.metrics.MeanAbsoluteError(),
                           tf.keras.metrics.MeanSquaredError()])
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
    callbacks=[early_stopping, reduce_lr], #reduce_lr  hist_callback
    validation_data=(X_val, y_val)
)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(1)[0]
print(best_hps)

print("START GET MODEL")
best_model = tuner.get_best_models(num_models=1)[0]

best_model.summary()

test_pred = best_model.predict(X_test)
import sklearn.metrics as metrics

r2 = metrics.r2_score(y_test, test_pred)
print("R2 Testing: ", r2)
rmse = np.sqrt(metrics.mean_squared_error(y_test, test_pred))
print('RMSD ( Root Mean Squared Error ) :', rmse)

print("## DONE ##")
