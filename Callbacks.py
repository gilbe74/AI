from neptune_optuna import NeptuneCallback
import tensorflow as tf

def myNeptuneCallback(run):
    neptune_cbk = NeptuneCallback(
        run=run,
        base_namespace="metrics",  # optionally set a custom namespace name
        log_model_diagram=True,
        log_on_batch=True
    )
    return neptune_cbk

def myEarlyStoppingCallback(metrics = 'val_loss', patience = 6):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=metrics,
                                                  patience=patience,
                                                  mode='min',
                                                  min_delta=1e-4,
                                                  restore_best_weights=True)
    return early_stopping

def myReduceLR(metrics = 'val_loss', factor = 0.1, patience = 2, min_lr = 1e-6):
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor=metrics,
                                         factor=factor,
                                         patience=patience,
                                         verbose=1,
                                         mode='min',
                                         # min_delta=0.0001,
                                         cooldown=0,
                                         min_lr=min_lr)
    return lr_callback

class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.val_loss = [1.3, 1.2, 1.1, 1.0]
        self.summing = 0

    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr, self.summing = self.schedule(epoch, lr, self.val_loss, self.summing)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.8f." % (epoch, scheduled_lr))

    def on_epoch_end(self, epoch, logs=None):
        self.val_loss[0] = self.val_loss[1]
        self.val_loss[1] = self.val_loss[2]
        self.val_loss[2] = self.val_loss[3]
        self.val_loss[3] = logs["val_loss"]
        print(
            "The average loss for epoch {} is {:7.2f} "
            "and mean absolute error is {:7.2f}.".format(
                epoch, logs["loss"], logs["mean_absolute_error"]
            )
        )

def lr_schedule_val(epoch, lr, val_loss, summing):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    counter = 0
    if(summing<=2):
        if (val_loss[3] >= val_loss[2]):
            counter += 1
            if (val_loss[2] >= val_loss[1]):
                counter += 1
                if (val_loss[1] >= val_loss[0]):
                    counter += 1
    else:
        if (val_loss[3] >= val_loss[2]):
            counter = 4


    if(summing ==0):
        if(counter==1):
            lr = lr * 0.1
            summing+=1
    elif(summing ==1):
        if (counter == 2):
            lr = lr * 0.05
            summing += 1
    elif (summing == 2):
        if (counter == 1):
            lr = lr * 0.05
            summing += 1
    elif (summing == 3):
        if (counter == 4):
            summing += 1
            lr = lr * 0.05
    elif (summing == 4):
        if (counter == 4):
            summing += 1
            lr = lr * 0.05

    return lr, summing

def callbacks(neptune = True, early = True, lr = True, run = None):
    callbacks = []
    if neptune and run != None:
        callbacks.append(myNeptuneCallback(run))
    if early:
        callbacks.append(myEarlyStoppingCallback())
    if lr:
        callbacks.append(myReduceLR())
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True))

    return callbacks