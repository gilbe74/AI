import warnings

import tensorflow as tf
import tensorflow.keras as keras
from optuna.integration import TFKerasPruningCallback, KerasPruningCallback
from optuna.integration.pytorch_lightning import Trainer
from optuna.trial import TrialState
from optuna.visualization import plot_optimization_history, plot_intermediate_values, plot_contour, \
    plot_param_importances
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from keras import Input, Model
from tensorflow.keras.layers import Dropout
import Utility as ut
import neptune.new as neptune
import optuna
import Callbacks as cb
import Models as md
warnings.simplefilter("ignore", UserWarning)

tags = ['LSTM_WS', 'Stacked']
run = neptune.init_run(
    project="gilberto.nobili/Optuna",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyNmNjMTI3Ni02ZjkwLTRiMTgtOGI5Zi03NzczMWJlODIwYTIifQ==",
    tags=tags
)  # your credentials

parameters = {
    "time_window": 40,
    "min_hidden_layers": 2,
    "max_hidden_layers": 2,
    "future_step": 1,
    "sampling": 1,
    "learning_rate": 7.2e-6,
    "learning_rate_tg": 7e-7,
    "n_epochs": 85,
    'dropout': 0,
    "label": 'Pitch',
    "val_split": 0,
    "max_trials": 4,
    "patience": 7,
    "filter_in": 'none',  # kalman wiener simple none
    "filter_out": 'none',  # kalman wiener simple none
    "optimizer": 'adam',  # adam
    "activation": 'tanh',  # tanh or relu or elu
    "scaler": 'Standard',  # Standard or MinMaxScaler or Normalizer
    "loss_function": 'huber_loss',  # huber_loss or mean_squared_error or log_cosh or mean_absolute_error
    "loss_metrics": 'val_loss' #val_loss or mean_absolute_error
}

ut.set_seed()

X_train, X_test, y_train, y_test = ut.clueanUpData(False, parameters['filter_in'], bestFeature=0)

X_train, X_test = ut.scalingData(X_train, X_test, parameters['scaler'])

X_train, X_test, y_train, y_test = ut.toSplitSequence(X_train, X_test, y_train, y_test, parameters['time_window'],
                                                      parameters['future_step'])

# activation = ut.get_activation(parameters['activation'])

# Number history timestamps
n_timestamps = X_train.shape[1]
# Number of input features to the model
n_features = X_train.shape[2]
# Number of output timestamps
n_future = y_train.shape[1]

DEFAULT_RETURN = 0.3


def objective(trial):
    tf.keras.backend.clear_session()

    # if parameters['max_hidden_layers'] > parameters['min_hidden_layers']:
    #     n_layers = trial.suggest_int("n_layers", parameters['min_hidden_layers'], parameters['max_hidden_layers'])
    # else:
    #     n_layers = parameters['max_hidden_layers']

    l1 = trial.suggest_categorical("kernel_regularizer_l1", [0.0, 0.01])
    l2 = trial.suggest_categorical("kernel_regularizer_l2", [0.0, 0.01])
    # l1 = 0.0
    # l2 = 0.0

    #batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    batch_size = 64#trial.suggest_int("batchsize", 32, 128, step=32, log=False)

    input_units = 256#trial.suggest_int("input_units", 256, 320, 64)
    # output_units = trial.suggest_int("output_units", 64, 128, 64)
    # dense_units = trial.suggest_int("dense_units", 32, 64, 32)

    # # model = tf.keras.models.Sequential()
    # # model.add(LSTM(units=input_units,
    # #                return_sequences=True,
    # #                activation=activation,
    # #                kernel_regularizer=L1L2(l1=l1, l2=l2),
    # #                #recurrent_dropout=recurrent_dropout,
    # #                input_shape=(X_train.shape[1], X_train.shape[2])))
    # # for i in range(n_layers):
    # #     num_hidden = trial.suggest_int(f'n_units_l{i}', 128, 256, 128)
    # #     model.add(LSTM(num_hidden, return_sequences=True))
    # #     # p = trial.suggest_float("dropout_l{}".format(i), 0.0, parameters['dropout'])
    # #     # if(p>0):
    # #     #     model.add(Dropout(p))
    # # # model.add(TimeDistributed(Dense(y_train.shape[1], activation='linear')))
    # # model.add(LSTM(units=output_units, return_sequences=False))
    # # model.add(Dense(units=dense_units, activation='linear'))
    # # # if(recurrent_dropout>0):
    # # #     model.add(Dropout(recurrent_dropout))
    # # model.add(Dense(y_train.shape[1], activation='linear'))
    #

    # activation_function = trial.suggest_categorical("activation", ['relu', 'elu', 'tanh'])
    # activation = ut.get_activation(activation_function)
    activation = ut.get_activation(parameters['activation'])

    input = tf.keras.layers.Input(shape=(n_timestamps, n_features))

    lstm1 = LSTM(input_units, return_sequences=True, return_state=True,
                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                 activation=activation)
    all_state_h, state_h, state_c = lstm1(input)
    states = [state_h, state_c]

    lstm2 = LSTM(input_units, return_sequences=False, activation=activation)
    all_state_h = lstm2(all_state_h, initial_state=states)

    # dropoutLast = Dropout(0.1)
    # all_state_h = dropoutLast(all_state_h)

    dense = Dense(n_future, activation='linear')
    output = dense(all_state_h)

    model = Model(input, output, name='model_LSTM_return_state')

    # from tensorflow.keras.layers import Bidirectional
    # model = tf.keras.models.Sequential()
    # model.add(Bidirectional(LSTM(input_units, activation=activation), input_shape=(n_timestamps, n_features)))
    # model.add(Dense(n_future, activation='linear'))

    # -------------------------------- LEARNING RATE -----------------------------
    # lr = 0.0001 #default
    lr = parameters['learning_rate']
    # lr = trial.suggest_float('learning_rate', 6e-6, 9e-6, log=True)
    # lr = trial.suggest_categorical("learning_rate", [6e-6, 8e-6])

    opti = ut.get_optimizer(optimizer=parameters['optimizer'],
                            learning_rate=lr)

    # loss_function = trial.suggest_categorical("loss", ['huber_loss', 'mean_squared_error','log_cosh'])
    model = md.compile_model(model, opti, parameters['loss_function'])

    # print(model.summary())

    my_callbacks = cb.callbacks(neptune=False,
                                early=parameters['patience'] > 0,
                                lr=True,
                                scheduler=False,
                                run=run,
                                opti=model.optimizer,
                                target=parameters['learning_rate_tg'],
                                patience=3,
                                loss=parameters['loss_metrics'])

    my_callbacks.append(TFKerasPruningCallback(trial,parameters['loss_metrics']))

    try:
        model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            shuffle=False,
            batch_size=batch_size,
            epochs=parameters['n_epochs'],
            verbose=True,
            callbacks=my_callbacks
        )
    except:

        print("An exception occurred on Model Fit")
        return DEFAULT_RETURN

    # Evaluate the model accuracy on the validation set.
    score = model.evaluate(X_test, y_test, verbose=1)
    return score[3] #3 MSE 2 MAE
    # return ut.model_performance(model)  # score[1]


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
study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
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

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

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
