import tensorflow as tf
import tensorflow.keras as keras
from optuna.integration import TFKerasPruningCallback, KerasPruningCallback
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
import neptune.new as neptune
from tensorflow.keras.regularizers import L1L2
import sklearn.metrics as metrics
import optuna

tags = ['LSTM_WS', 'TEST']
run = neptune.init_run(
    project="gilberto.nobili/Optuna",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyNmNjMTI3Ni02ZjkwLTRiMTgtOGI5Zi03NzczMWJlODIwYTIifQ==",
    tags=tags
)  # your credentials

parameters = {
    "time_window": 40,
    "min_hidden_layers": 0,
    "max_hidden_layers": 0,
    "future_step": 1,
    "sampling": 1,
    "n_epochs": 20,
    'dropout': 0,
    "label": 'Pitch',
    "val_split": 0,
    "max_trials": 40,
    "scaler": 'Standard',  # Standard or MinMaxScaler or Normalizer
    "loss_function": 'huber_loss'  # huber_loss or mean_squared_error
}

DEFAULT_RETURN = 0.1

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
elif parameters['scaler'] == 'Normalizer':
    from sklearn.preprocessing import Normalizer
    scaler = Normalizer()
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


def model_performance(model, X=X_val, y=y_val):
    """
    Get accuracy score on validation/test data from a trained model
    """
    try:
        y_pred = model.predict(X)
        y_test = y_val[:, 0]
        r2 = metrics.r2_score(y_test, y_pred)
        print("R2 Testing: ", r2)
        return round(r2, 3)
    except:
        print("R2 Testing FAILS")
        return DEFAULT_RETURN


# 1. Define an objective function to be maximized.
def objective(trial):
    if parameters['max_hidden_layers'] > parameters['min_hidden_layers']:
        n_layers = trial.suggest_int("n_layers", parameters['min_hidden_layers'], parameters['max_hidden_layers'])
    else:
        n_layers = parameters['max_hidden_layers']

    lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

    l1 = trial.suggest_categorical("kernel_regularizer_l1", [0.0, 0.01])
    l2 = trial.suggest_categorical("kernel_regularizer_l2", [0.0, 0.01])

    batch_size = 64 #trial.suggest_categorical("batch_size", [32, 64])
    # epochs = trial.suggest_int("epochs", 10, parameters['n_epochs'], step=5, log=False)
    # batchsize = trial.suggest_int("batchsize", 32, 128, step=32, log=False)

    if parameters['dropout'] > 0:
        recurrent_dropout = trial.suggest_float('recurrent_dropout', 0.0, parameters['dropout'], log=True)
    else:
        recurrent_dropout = 0.0

    input_units = trial.suggest_int("input_units", 32, 512, 32)
    output_units = trial.suggest_int("output_units", 32, 512, 32)

    model = tf.keras.models.Sequential()
    model.add(LSTM(units=input_units,
                   return_sequences=True,
                   kernel_regularizer=L1L2(l1=l1, l2=l2),
                   recurrent_dropout=recurrent_dropout,
                   input_shape=(X_train.shape[1], X_train.shape[2])))
    for i in range(n_layers):
        num_hidden = trial.suggest_int(f'n_units_l{i}', 32, 512, 32)
        model.add(LSTM(num_hidden, return_sequences=True))
        # p = trial.suggest_float("dropout_l{}".format(i), 0.0, parameters['dropout'])
        # if(p>0):
        #     model.add(Dropout(p))
    model.add(LSTM(units=output_units))
    model.add(Dense(y_train.shape[1], activation='linear'))

    model.compile(loss='huber_loss', optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(),
                           tf.keras.metrics.MeanSquaredError()])

    try:
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            shuffle=True,
            batch_size=batch_size,
            epochs=parameters['n_epochs'],
            verbose=True,
            callbacks=[TFKerasPruningCallback(trial, "val_loss")]
        )
    except:
        print("An exception occurred on Model Fit")
        return DEFAULT_RETURN

    # Evaluate the model accuracy on the validation set.
    # score = model.evaluate(X_val, y_val, verbose=1)
    return model_performance(model)  # score[1]


import neptune.new.integrations.optuna as optuna_utils

neptune_callback = optuna_utils.NeptuneCallback(run,
                                                log_plot_slice=False,  # do not create/log plot_slice
                                                log_plot_contour=False,  # do not create/log plot_contour
                                                log_plot_parallel_coordinate=False,
                                                log_plot_intermediate_values=False,
                                                log_plot_optimization_history=False,
                                                log_plot_param_importances=False
                                                )

# 3. Create a study object and optimize the objective function.
# study = optuna.create_study(direction='maximize', pruner=optuna.pruners.SuccessiveHalvingPruner())
study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
print(f"Sampler is {study.sampler.__class__.__name__}")
study.optimize(objective, n_trials=parameters['max_trials'], callbacks=[neptune_callback])

print("Number of finished trials: {}".format(len(study.trials)))

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("### Best trial ####")
trial = study.best_trial
print("  Value: {}".format(trial.value))

# best = study.best_params
# for x in best:
#     print(x)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# best_model = create_model(study.best_trial)
# best_model.fit(X_train, y_train)
# print("Performance: ", model_performance(best_model))

print("## DONE ##")

fig = plot_optimization_history(study)
fig.show()
run["evaluation/plot_optimization_history"].upload(fig)

try:
    fig = plot_intermediate_values(study)
    fig.show()
    run["evaluation/plot_intermediate_values"].upload(fig)
except:
    print("An exception occurred plot_intermediate_values")

try:
    fig = plot_contour(study)
    fig.show()
    run["evaluation/plot_contour"].upload(fig)
except:
    print("An exception occurred plot_contour")

try:
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()
    run["evaluation/plot_param_importances"].upload(fig)
except:
    print("An exception occurred plot_param_importances")

try:
    fig = optuna.visualization.plot_slice(study)
    fig.show()
    run["evaluation/plot_slice"].upload(fig)
except:
    print("An exception occurred plot_slice")

try:
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.show()
    run["evaluation/plot_parallel_coordinate"].upload(fig)
except:
    print("An exception occurred plot_parallel_coordinate")

run.stop()
