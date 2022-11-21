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
from pandas import DataFrame , concat
from sklearn.metrics import mean_absolute_error , mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from numpy import mean , concatenate
from numpy import array, hstack
from pickle import dump,load
import Utility as ut
import warnings
warnings.simplefilter("ignore", UserWarning)
np.random.seed(19740429)
tf.random.set_seed(51)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
from scipy import signal

DISPLAY = False

parameters = {
    "debug": False,
    "time_window": 40,
    "layers": [384, 256, 384, 320, 128],
    "future_step": 1,
    "sampling": 1,
    "learning_rate": 0.001,
    "batch_size": 64,
    "n_epochs": 50,
    'dropout': 0,
    "label": 'Pitch',
    "patience": 3,
    "val_split": 0,
    "stateful": False,
    "scaler": 'MinMaxScaler', #Standard or MinMaxScaler
    "loss_function": 'huber_loss'  # huber_loss or mean_squared_error
}
tags = ['LSTM_BO']


import DataRetrive as es
data = es.retriveDataSet(False)

if(data is None):
    data = es.getDataSet()
    es.saveDataSet(data)
if(data.empty):
    print('Is the DataFrame empty!')
    raise SystemExit

# ['Datetime','SinkMin_AP','YawRate','Bs','Heel','Pitch', 'Lwy', 'Tws']
data=data.drop(['Datetime','Bs','YawRate','Heel', 'Lwy','Tws', 'SinkMin_AP'], axis=1)
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

data.info()

TEST_SPLIT = 1 - (test_index / len(Xi)) # Test Split to last racing day

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xi, yi, test_size=TEST_SPLIT, shuffle=False)

# Split Validation Set
if parameters['val_split'] == 0:
    X_val = X_test
    y_val = y_test
else:
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=parameters['val_split'], shuffle=False) # 0.25 x 0.8 = 0.2

#Scale the 3 dataset
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


#FreeUp some memeory
del(dataset)
del(data)
del(Xi)
del(yi)
del(zi)


n_features = X_train.shape[2]
n_label = y_train.shape[1]

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=parameters['patience'],
                                                  mode='min',
                                                  restore_best_weights=True)

def build_model(hp):
    # initialize the learning rate choices and optimizer
    lr = hp.Choice("learning_rate",
                   values=[1e-2, 1e-3, 1e-4])
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    model = tf.keras.models.Sequential()
    #model.add(Dense(X_train.shape[1], activation='linear'))
    model.add(LSTM(hp.Int('input_unit',min_value=32,max_value=512,step=32), return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    # model.add(LSTM(32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    for i in range(hp.Int('n_layers', 0, 1)):
        model.add(LSTM(hp.Int(f'lstm_{i}_units',min_value=32,max_value=512,step=32), return_sequences=True))
    model.add(LSTM(hp.Int('layer_2_neurons',min_value=32,max_value=512,step=32)))
    # model.add(keras.layers.Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.1,step=0.1)))
    model.add(Dense(y_train.shape[1], activation='linear'))


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=parameters['loss_function'],
                      metrics=[tf.keras.metrics.RootMeanSquaredError(),
                               tf.keras.metrics.MeanAbsoluteError(),
                               tf.keras.metrics.MeanSquaredError(),
                               'accuracy'])
    return model


class MyTuner(keras_tuner.tuners.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 128, step=32)
        return super(MyTuner, self).run_trial(trial, *args, **kwargs)

tuner = MyTuner(
    build_model,
    objective='mean_squared_error',
    max_trials=100,
    executions_per_trial=1,
    overwrite = True
)

# tuner = BayesianOptimization(
#     build_model,
#     objective='mean_squared_error',
#     max_trials=40,
#     executions_per_trial=1,
#     overwrite=True
# )

tuner.search(
    x=X_train,
    y=y_train,
    epochs=20,
    batch_size=parameters['batch_size'],
    callbacks=[early_stopping],
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