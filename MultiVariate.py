import matplotlib
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
from numpy import array
from pickle import dump,load
import warnings
warnings.simplefilter("ignore", UserWarning)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from neptune.new.types import File
import glob
import pydot
import absl.logging
import Utility as ut
import DataRetrive as es
absl.logging.set_verbosity(absl.logging.ERROR)
import errno

def set_seed(sd=19740429):
    from numpy.random import seed
    import random as rn
    ## numpy random seed
    seed(sd)
    ## core python's random number
    rn.seed(sd)
    ## tensor flow's random number
    tensorflow.random.set_seed(sd)

set_seed()

NEPTUNE = True
DEBUG = False
PRINT = True
DISPLAY = True
PRINT_FULLDATA = False
NEPTUNE_PROJECT_NAME = "gilberto.nobili/Pitch"
NEPTUNE_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyNmNjMTI3Ni02ZjkwLTRiMTgtOGI5Zi03NzczMWJlODIwYTIifQ=="

if(DEBUG):
    parameters = {
        "debug": True,
        "time_window": 6,
        "future_step": 1,
        "sampling": 1,
        "learning_rate": 0.1,
        "batch_size": 1,
        "n_epochs": 2,
        "label": 'Pitch',
        "patience": 0,
        "val_split": 0,
        "scaler": 'MinMaxScaler',  # Standard or MinMaxScaler
        "loss_function": 'huber_loss'  # huber_loss or mean_squared_error
    }
    tags = ['LSTM_WS', 'DEBUG']
else:
    parameters = {
        "debug": False,
        "time_window": 30,
        "layers": [384, 256, 384, 320, 128],
        "future_step": 1,
        "sampling": 1,
        "learning_rate": 0.001,
        "batch_size": 64,
        "n_epochs": 20,
        'dropout': 0,
        "label": 'Pitch',
        "patience": 0,
        "val_split": 0,
        "stateful": False,
        "scaler": 'MinMaxScaler', #Standard or MinMaxScaler
        "loss_function": 'huber_loss'  # huber_loss or mean_squared_error
    }
    tags = ['LSTM_WS']

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

# remove from the DataFrame the colum of the label
data = data.drop(parameters['label'], axis=1)
Xi = data.values
yi = yi.reshape((len(yi), 1))

data.info()


TEST_SPLIT = 1 - (test_index / len(Xi)) # Test Split to last racing day

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xi, yi, test_size=TEST_SPLIT, shuffle=False)

if(PRINT_FULLDATA):
    import matplotlib.pyplot as pltDataSet
    range_history = len(yi)
    range_future = len(y_test)
    range_forecast = list(range(test_index, range_history))
    pltDataSet.figure(figsize=(10, 5))
    pltDataSet.plot(np.arange(range_history), np.array(yi), label=parameters['label'])
    pltDataSet.plot(np.arange(len(data['Day'].values)), data['Day'].values, label='Days')
    pltDataSet.plot(np.arange(range_history), ut.downsampling(yi, 50), label='Human Action', color='red', alpha=0.5)
    # pltDataSet.plot(np.arange(len(data['isSailing'].values)), data['isSailing'].values*10, label='Sailing')
    # pltDataSet.plot(np.arange(len(data['isTowing'].values)), data['isTowing'].values*5, label='Towing')
    pltDataSet.plot(range_forecast, np.array(y_test), label='TestSet')
    pltDataSet.title("Full DataSet")
    pltDataSet.xlabel('Time step' ,  fontsize=18)
    pltDataSet.legend(loc='upper right')
    pltDataSet.ylabel('Values', fontsize=18)
    pltDataSet.legend()
    pltDataSet.show()

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

if(NEPTUNE):
    run = neptune.init_run(
        project=NEPTUNE_PROJECT_NAME,
        tags=tags,
        api_token=NEPTUNE_TOKEN,
    )  # your credentials
    run["model/parameters"] = parameters

from time import time
start_at = time()
print("## START ##")

from tensorflow.keras import layers
# # define model
def create_uncompiled_model():
    tmp_model = tf.keras.models.Sequential()
    if parameters['debug']:
        tmp_model.add(LSTM((n_features + 1),
                       return_sequences=True,
                       input_shape=(X_train.shape[1], X_train.shape[2])))
        tmp_model.add(LSTM((int)((n_features + n_label) * 2 / 3)))
        tmp_model.add(Dense(n_label, activation='linear'))
    else:
        tmp_model.add(LSTM(parameters['layers'][0], return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), stateful=parameters['stateful']))
        for i in range(1, len(parameters['layers'])-1, 1):
            tmp_model.add(LSTM(parameters['layers'][i], return_sequences=True))  # i
        tmp_model.add(LSTM(parameters['layers'][len(parameters['layers'])-1]))  # Layer 2
        if parameters['dropout']>0:
            tmp_model.add(Dropout(parameters['dropout']))
        tmp_model.add(Dense(y_train.shape[1], activation='linear'))
    return tmp_model


def create_model():
    tmp_model = create_uncompiled_model()

    tmp_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=parameters['learning_rate']),
                  loss=parameters['loss_function'],
                  metrics=[tf.keras.metrics.RootMeanSquaredError(),
                           tf.keras.metrics.MeanAbsoluteError(),
                           tf.keras.metrics.MeanSquaredError(),'accuracy'])

    # Log model summary
    if (NEPTUNE):
        tmp_model.summary(print_fn=lambda x: run['model/summary'].log(x))

    print(tmp_model.summary())

    return tmp_model

model = create_model()

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=parameters['patience'],
                                                  mode='min',
                                                  restore_best_weights=True)

if NEPTUNE:
    neptune_cbk = NeptuneCallback(
        run=run,
        base_namespace="metrics",  # optionally set a custom namespace name
        log_model_diagram=True,
        log_on_batch=True,
    )

    if parameters['patience'] > 0:
        my_callbacks = [early_stopping, neptune_cbk]
    else:
        my_callbacks = [neptune_cbk]
else:
    if parameters['patience'] > 0:
        my_callbacks = [early_stopping]
    else:
        my_callbacks = []


history = model.fit(X_train, y_train,
                epochs=parameters['n_epochs'],
                batch_size=parameters['batch_size'],
                validation_data=(X_val, y_val),
                callbacks=my_callbacks)


    # if (NEPTUNE):
    #     neptune_cbk = NeptuneCallback(
    #         run=run,
    #         base_namespace="metrics",  # optionally set a custom namespace name
    #         log_model_diagram=True,
    #         log_on_batch=True,
    #     )
    #     history = model.fit(X_train, y_train,
    #                         epochs=parameters['n_epochs'],
    #                         batch_size=parameters['batch_size'],
    #                         validation_data=(X_val, y_val),
    #                         callbacks=[early_stopping, neptune_cbk])
    # else:
    #     history = model.fit(X_train, y_train,
    #                         epochs=parameters['n_epochs'],
    #                         batch_size=parameters['batch_size'],
    #                         validation_data=(X_val, y_val),
    #                         callbacks=[early_stopping])



print("--- %s seconds ---" % int(time() - start_at))

if (NEPTUNE):
    model.save('trained_model', overwrite=True)
    run['model/weights/saved_model'].upload('trained_model/saved_model.pb')
    for name in glob.glob('trained_model/variables/*'):
        run[name].upload(name)

if(PRINT):
    print("Valuate the model")

    SINGLEPOINT = int(len(X_test) / 2)

    # # demonstrate prediction
    # val_results = model.evaluate(X_val, y_val)
    # val_pred = model.predict(X_val)
    # print("Validation loss, acc:", val_results)

    # Evaluate the model on the test data using `evaluate`
    test_results = model.evaluate(X_test, y_test, batch_size=parameters['batch_size'])
    # test_results = model.evaluate(X_test, y_test, batch_size=1)
    if (NEPTUNE):
        for j, metric in enumerate(test_results):
            run['test/scores/{}'.format(model.metrics_names[j])] = metric

    # Log predictions as table
    test_pred = model.predict(X_test)
    print("Test loss, rmse, mae, mse:", test_results)

    print("Error Valutation")
    import math
    def distance(a, b):
        if (a == b):
            return 0
        elif (a < 0) and (b < 0) or (a > 0) and (b >= 0):  # fix: b >= 0 to cover case b == 0
            if (a < b):
                return (abs(abs(a) - abs(b)))
            else:
                return -(abs(abs(a) - abs(b)))
        else:
            return math.copysign((abs(a) + abs(b)), b)

    import matplotlib.pyplot as pltError
    error_scores = list()
    for i in range(len(test_pred)):
        real = y_test[i].item(0)
        expect = test_pred[i].item(0)
        delta = distance(real,expect)
        error_scores.append(delta)
    delta_array = np.array(error_scores)
    delta_df = pd.DataFrame(delta_array)
    description = delta_df.describe()
    print(delta_df.describe())
    fig = pltError.figure(figsize=(10, 7))
    # pltError.title("Error Description")
    pltError.grid()
    pltError.boxplot(delta_array)
    pltError.draw()
    pltError.pause(0.1)
    if (NEPTUNE):
        run["evaluation/prediction"].upload(fig)

    SENSOR_ERROR = 0.05

    range_history = len(y_test)
    range_future = len(test_pred)
    # start = int(len(test_pred)/2)
    range_forecast = list(range(SINGLEPOINT, SINGLEPOINT + parameters['future_step']))
    y_test = y_test[:, 0]
    y_test_pred_single = test_pred[:, 0]

    # Generate predictions (probabilities -- the output of the last layer)
    print("Generate predictions for 1 samples - ", SINGLEPOINT)
    predictions = model.predict(X_test[SINGLEPOINT:SINGLEPOINT + 1])
    print('>Expected=%.2f, Predicted=%.2f' % (y_test[SINGLEPOINT:SINGLEPOINT + 1], predictions))
    # print("Predictions for index: ", start, "Model Out:", predictions, "Real Out: ", y_test[start:start + 1])

    # model.reset_states()
    # re-define model
    # n_batch = 1
    # new_model = tf.keras.models.Sequential()
    # new_model.add(LSTM(n_features + 1, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    # new_model.add(LSTM((int)((n_features + 2) * 2 / 3)))
    # new_model.add(Dense(n_steps_out, activation='linear'))
    # # copy weights
    # old_weights = model.get_weights()
    # new_model.set_weights(old_weights)
    # # compile model
    # new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mean_squared_error")
    # # new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=tf.keras.losses.Huber())
    # yhat = new_model.predict(X_test[start:start + 1])
    # print('>Expected=%.2f, Predicted=%.2f' % (y_test[start:start + 1], yhat))

    import matplotlib.pyplot as pyplotMetrics
    fig, ax = pyplotMetrics.subplots(2,1)
    pyplotMetrics.ion()
    # pyplotMetrics.subplot(211)
    ax[0].plot(history.history['loss'], label='train')
    ax[0].plot(history.history['val_loss'], label='validation')
    ax[0].set_ylabel("Loss", fontsize=10)
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    # Root Mean Squared Error
    ax[1].plot(history.history['root_mean_squared_error'], label='train')
    ax[1].plot(history.history['val_root_mean_squared_error'], label='validation')
    ax[1].legend(loc='upper right')
    ax[1].set_ylabel("RMSE", fontsize=10)
    ax[1].set_xlabel("Epochs", fontsize=10)
    ax[1].grid(True)
    pyplotMetrics.draw()
    pyplotMetrics.pause(0.1)
    if(NEPTUNE):
        run["evaluation/metrics"].upload(fig)

    # R2
    import sklearn.metrics as metrics
    r2 = metrics.r2_score(y_test, test_pred)
    print("R2 Testing: ", r2)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, test_pred))
    print('RMSD ( Root Mean Squared Error ) :', rmse)

    if (NEPTUNE):
        run["model/parameters/n_epochs"] = len(history.history['loss'])
        run["test/scores/R2"] = r2
        run["test/scores/expected"] = y_test[SINGLEPOINT:SINGLEPOINT + 1]
        run["test/scores/predicted"] = predictions



    import matplotlib.pyplot as pltCoere
    # Fixing random state for reproducibility
    # Two signals with a coherent part at 10Hz and a random part
    s1 = np.array(y_test_pred_single)
    s2 = np.array(y_test)
    fig, axs = pltCoere.subplots(2, 1, sharex=True)
    pltCoere.ion()
    axs[0].plot(np.arange(range_history), np.array(y_test), label='Real', color='#1f77b4') #blue
    axs[0].plot(np.arange(range_future), np.array(y_test_pred_single), label='Prediction LSTM', color='#ff7f0e',
                alpha=0.8)
    if parameters['future_step'] > 1:
        axs[0].plot(range_forecast, np.array(test_pred[SINGLEPOINT]), label='Forecasted with LSTM', color='red')
    else:
        axs[0].scatter(SINGLEPOINT, predictions, color="red", marker="x", s=70, label="Single")
    axs[0].legend(loc='upper right')
    # axs[0].set_xlabel('Time step' ,  fontsize=18)
    axs[0].set_ylabel(parameters['label'], fontsize=10)
    axs[0].legend(loc='upper right')
    axs[0].grid(True)
    # cxy, f = axs[1].cohere(s1, s2, 256, 1. / dt)
    axs[1].xcorr(s1, s2, usevlines=True, maxlags=10, normed=True, lw=2)
    axs[1].set_ylabel('coherence', fontsize=10)
    # adding grid to the graph
    x = np.arange(range_future)
    markerline, stemlines, baseline = pltCoere.stem(x, delta_df, '-', markerfmt=' ')
    pltCoere.setp(baseline, color='r', linewidth=2)
    axs[1].axhline(SENSOR_ERROR, color="red", linewidth=0.5)
    axs[1].axhline(-SENSOR_ERROR, color="red", linewidth=0.5)
    axs[1].grid(True)
    axs[1].axhline(0, color='green', lw=2)
    axs[1].set_xlabel('Time step',  fontsize=10)
    axs[0].autoscale(axis='y', enable=True)
    axs[1].autoscale(axis='y', enable=True)
    fig.tight_layout()
    pltCoere.draw()
    pltCoere.pause(0.1)
    if(NEPTUNE):
        run["evaluation/prediction"].upload(fig)


    def zoom_in(myplot, delta, gap): #3000 = 2.5 minutes
        x_min = SINGLEPOINT - delta
        x_max = SINGLEPOINT + delta
        axs[0].set_xlim([x_min, x_max])
        visible_y_min, visible_y_max = ut.minMax(s2,y_test_pred_single, x_min, x_max, gap)
        axs[0].set_ylim([visible_y_min, visible_y_max])
        axs[1].set_xlim([x_min, x_max])
        series = delta_df.to_numpy()
        visible_y_min = series[x_min: x_max].min()
        visible_y_min = visible_y_min - abs(visible_y_min) * gap
        visible_y_max = series[x_min: x_max].max()
        visible_y_max = visible_y_max + abs(visible_y_max) * gap
        axs[1].set_ylim([visible_y_min, visible_y_max])
        pltCoere.draw()
        pltCoere.pause(0.1)
        if (NEPTUNE):
            run["evaluation/prediction_zoom_" + str(delta)].upload(fig)

    zoom_in(pltCoere, 3000, 0.1) #5 minutes
    zoom_in(pltCoere, 200, 0.1) #20 seconds

    #Plot Avg
    axs[0].plot(np.arange(range_history), ut.downsampling(y_test, 10), label='Human', color='yellow', alpha=0.8)
    axs[0].plot(np.arange(range_future), ut.downsampling(y_test_pred_single, 10), label='Slow', color='pink', alpha=0.8)

    zoom_in(pltCoere, 60, 0.1) #6 seconds

    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.ion()
    sns.histplot(delta_df, color="red", label="Errors", kde=True, stat="density", linewidth=3, bins=int(180 / 5))
    # sns.distplot(delta_df, hist=Truehist=True, kde=True,
    #              bins=int(180 / 5), color='darkblue',
    #              hist_kws={'edgecolor': 'black'},
    #              kde_kws={'linewidth': 3});
    plt.title("Error Distribution " + parameters['label'])
    plt.draw()
    plt.pause(0.1)
    if (NEPTUNE):
        run["evaluation/error"].upload(fig)

if(NEPTUNE):
    run.stop()


#Plotting
if(PRINT):
    if(DISPLAY):
        pyplotMetrics.show(block=True)
        pltCoere.show(block=True)
        plt.show(block=True)
        pltError.show(block=True)
    plt.close()
    pltCoere.close()
    pyplotMetrics.close()
    pltError.close()

print("## DONE ##")