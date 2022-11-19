import tensorflow as tf
import tensorflow.keras as keras
from optuna.integration import TFKerasPruningCallback
from optuna.integration.pytorch_lightning import Trainer
from optuna.trial import TrialState
# from optuna.visualization.matplotlib import plot_optimization_history, plot_intermediate_values, plot_contour,
#     plot_param_importances
from optuna.visualization import plot_optimization_history, plot_intermediate_values, plot_contour, \
    plot_param_importances
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import Utility as ut
import DataRetrive as es
import numpy as np
import pandas as pd

import optuna

parameters = {
        "time_window": 3,
        "min_hidden_layers": 0,
        "max_hidden_layers": 1,
        "future_step": 1,
        "sampling": 1,
        "n_epochs": 3,
        'dropout': 0,
        "label": 'Pitch',
        "val_split": 0,
        "max_trials": 4,
        "scaler": 'MinMaxScaler', #Standard or MinMaxScaler
        "loss_function": 'huber_loss'  # huber_loss or mean_squared_error
    }

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

if parameters['sampling'] > 1:
    data = data.rolling(parameters['sampling']).mean()
    data = data.iloc[::parameters['sampling'], :]
    data.dropna(inplace=True)

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

# data.info()


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

# 1. Define an objective function to be maximized.
def objective(trial):

    n_layers = trial.suggest_int("n_layers", parameters['min_hidden_layers'], parameters['max_hidden_layers'])

    if parameters['dropout']>0 :
        recurrent_dropout = trial.suggest_float('recurrent_dropout', 0.0, parameters['dropout'], log=True)
    else:
        recurrent_dropout = 0.0

    model = tf.keras.models.Sequential()
    model.add(LSTM(units=trial.suggest_int("units", 16, 384, 16),
                   return_sequences=True,
                   recurrent_dropout= recurrent_dropout,
                   input_shape=(X_train.shape[1], X_train.shape[2])))
    for i in range(n_layers):
        num_hidden = trial.suggest_int(f'n_units_l{i}', 32, 512, 32)
        model.add(LSTM(num_hidden, return_sequences=True))
        # p = trial.suggest_float("dropout_l{}".format(i), 0.0, parameters['dropout'])
        # if(p>0):
        #     model.add(Dropout(p))
    model.add(LSTM(units=trial.suggest_int("units_last", 16, 384, 16)))
    model.add(Dense(X_train.shape[1], activation='linear'))

    # We compile our model with a sampled learning rate.
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size",[32,64,128])
    model.compile(loss='huber_loss', optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError(),'accuracy'])

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        shuffle=True,
        batch_size=batch_size,
        epochs= parameters['n_epochs'],
        verbose=True,
        callbacks=[TFKerasPruningCallback(trial, "val_loss")],
    )

    # Evaluate the model accuracy on the validation set.
    score = model.evaluate(X_val, y_val, verbose=1)
    return score[1]


# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
print(f"Sampler is {study.sampler.__class__.__name__}")
study.optimize(objective, n_trials=parameters['max_trials'])

print("Number of finished trials: {}".format(len(study.trials)))

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("### Best trial ####")
trial = study.best_trial
print("  Value: {}".format(trial.value))

best = study.best_params
for x in best:
    print (x)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

print("## DONE ##")

fig = plot_optimization_history(study)
fig.show()

fig = plot_intermediate_values(study)
fig.show()

fig = plot_contour(study)
fig.show()

# fig = optuna.visualization.plot_pareto_front(study)
# fig.show()

fig = optuna.visualization.plot_param_importances(study)
fig.show()
# plot_param_importances(study)