import tensorflow as tf
import warnings

from neptune.common.deprecation import NeptuneDeprecationWarning

from LRFinder import LRFinder

warnings.simplefilter("ignore", UserWarning)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
import glob
import absl.logging
import Utility as ut
import Callbacks as cb
import Models as md
absl.logging.set_verbosity(absl.logging.ERROR)
warnings.simplefilter("ignore", NeptuneDeprecationWarning)

SAVE_SCORE = True
NEPTUNE = True
PRINT = True
DISPLAY = True
PRINT_FULLDATA = False
NEPTUNE_PROJECT_NAME = "gilberto.nobili/Pitch"
NEPTUNE_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyNmNjMTI3Ni02ZjkwLTRiMTgtOGI5Zi03NzczMWJlODIwYTIifQ=="

parameters = {
    "debug": False,
    "time_window": 80,
    "layers": [384,128],
    "future_step": 1,
    "sampling": 1,
    "learning_rate": 1.7e-5,
    "learning_rate_tg": 8e-7,
    "l1": 0.0,
    "l2": 0.0,
    "batch_size": 64,
    "n_epochs": 50,
    'dropout': 0,
    "label": 'Pitch',
    "patience": 6,
    "filter_in": 'none',  # kalman wiener simple none
    "filter_out": 'none',  # kalman wiener simple none
    "optimizer": 'adam',  # adam
    "activation": 'tanh', # tanh or relu or elu
    "scaler": 'Standard',  # Standard or MinMaxScaler or Normalizer or Robust or MaxAbsScaler
    "loss_function": 'huber_loss'  # huber_loss or mean_squared_error
}
tags = ['LSTM_WS', 'TW=80', "ReturnState", "BS=64", "Shuffle"]

ut.set_seed()

X_train, X_test, y_train, y_test = ut.clueanUpData(False, parameters['filter_in'], bestFeature = 0)

X_train, X_test = ut.scalingData(X_train, X_test, parameters['scaler'])

X_train, X_test, y_train, y_test = ut.toSplitSequence(X_train, X_test, y_train, y_test, parameters['time_window'], parameters['future_step'])

if(NEPTUNE):
    run = neptune.init_run(
        project=NEPTUNE_PROJECT_NAME,
        tags=tags,
        api_token=NEPTUNE_TOKEN,
    )  # your credentials
    run["model/parameters"] = parameters

n_timestamps = X_train.shape[1]
# Number of output timestamps
n_future = y_train.shape[1]
# Number of input features to the model
n_features = X_train.shape[2]

def create_model():
    tf.keras.backend.clear_session()

    activation = ut.get_activation(parameters['activation'])

    tmp_model = md.create_uncompiled_model_ReturnState_ReturnSeq_2Layers(n_timestamps,n_future,n_features,256,activation)

    opti = ut.get_optimizer(optimizer=parameters['optimizer'],
                            learning_rate=parameters['learning_rate'])

    tmp_model = md.compile_model(tmp_model, opti,parameters['loss_function'])

    # Log model summary
    if (NEPTUNE):
        tmp_model.summary(print_fn=lambda x: run['model/summary'].log(x))

    print(tmp_model.summary())

    return tmp_model

model = create_model()

# ut.LearningRatePlot(model,X_train,y_train, 2, parameters['batch_size'])

my_callbacks = cb.callbacks(neptune=False,
                            early = parameters['patience'] > 0,
                            lr = True,
                            scheduler= False,
                            run = run,
                            opti = model.optimizer,
                            target= parameters['learning_rate_tg'],
                            patience = 3)
def myNeptuneCallback(run):
    neptune_cbk = NeptuneCallback(
        run=run,
        base_namespace="metrics",  # optionally set a custom namespace name
        log_model_diagram=True,
        log_on_batch=True
    )
    return neptune_cbk
my_callbacks.append(myNeptuneCallback(run))

from time import time
start_at = time()
print("## START ##")

history = model.fit(X_train, y_train,
                epochs=parameters['n_epochs'],
                batch_size=parameters['batch_size'],
                validation_data=(X_test, y_test),
                shuffle=True,
                callbacks=my_callbacks
                )

print("--- %s seconds ---" % int(time() - start_at))

if (NEPTUNE):
    model.save('trained_model', overwrite=True)
    run['model/weights/saved_model'].upload('trained_model/saved_model.pb')
    for name in glob.glob('trained_model/variables/*'):
        run[name].upload(name)

if(PRINT):
    import PlotResults as pl
    pl.PlotResult(model, X_test, y_test, history, run=run, batch_size=parameters['batch_size'], n_future=n_future, saveDelta=True, SENSOR_ERROR=0.05)

if(NEPTUNE):
    run.stop()


print("## DONE ##")