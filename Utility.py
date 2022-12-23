import math
import numpy as np
import pandas as pd
from moto import es
from numpy import array
from numpy import minimum
import tensorflow as tf
import sklearn.metrics as metrics
from tensorflow.keras.activations import elu, relu, tanh
from tensorflow.python.keras.activations import sigmoid


def set_seed(sd=19740429):
    from numpy.random import seed
    import random as rn
    ## numpy random seed
    seed(sd)
    ## core python's random number
    rn.seed(sd)
    ## tensor flow's random number
    tf.random.set_seed(sd)

#dow-sample column based on window size
def downsampling(dfi, window):
    df = pd.DataFrame(dfi, columns=['x1'])
    df["x1"] = df["x1"].rolling(window=window).mean()
    df["x2"] = np.nan
    df.loc[window - 1::window, "x2"] = df["x1"]
    df["x2"].fillna(method="ffill", inplace=True)
    # df.dropna(inplace=True)
    return np.array(df["x2"])

# split a multivariate sequence into samples
def split_sequences(sequences,
                    n_steps_in,
                    n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x = sequences[i:end_ix, :-1]
        seq_y = sequences[end_ix-1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def minMax(array1,array2, xmin, xmax, gap):
    y_min1 = array1[xmin:xmax].min()
    y_min2 = array2[xmin:xmax].min()
    y_min = minimum(y_min1, y_min2)
    y_min = y_min - abs(y_min) * gap

    y_max1 = array1[xmin:xmax].max()
    y_max2 = array2[xmin:xmax].max()
    y_max = max(y_max1, y_max2)
    y_max = y_max + abs(y_min) * gap

    return y_min, y_max

def fft_denoiser(x, n_components, to_real=True):
    """Fast fourier transform denoiser.

    Denoises data using the fast fourier transform.

    Parameters
    ----------
    x : numpy.array
        The data to denoise.
    n_components : int
        The value above which the coefficients will be kept.
    to_real : bool, optional, default: True
        Whether to remove the complex part (True) or not (False)

    Returns
    -------
    clean_data : numpy.array
        The denoised data.

    References
    ----------
    .. [1] Steve Brunton - Denoising Data with FFT[Python]
       https://www.youtube.com/watch?v=s2K1JfNR7Sc&ab_channel=SteveBrunton

    """
    n = len(x)

    # compute the fft
    fft = np.fft.fft(x, n)

    # compute power spectrum density
    # squared magnitud of each fft coefficient
    PSD = fft * np.conj(fft) / n

    # keep high frequencies
    _mask = PSD > n_components
    fft = _mask * fft

    # inverse fourier transform
    clean_data = np.fft.ifft(fft)

    if to_real:
        clean_data = clean_data.real

    return clean_data

def kalman(x):
    from kalmankit import KalmanFilter
    A = np.expand_dims(np.ones((len(x), 1)), axis=1)  # transition matrix
    xk = np.array([[1]])  # initial mean estimate

    Pk = np.array([[1]])  # initial covariance estimate
    Q = np.ones((len(x))) * 0.005  # process noise covariance

    H = A.copy()  # observation matrix
    R = np.ones((len(x))) * 0.01  # measurement noise covariance

    # run Kalman filter
    kf = KalmanFilter(A=A, xk=xk, B=None, Pk=Pk, H=H, Q=Q, R=R)
    states, errors = kf.filter(Z=x, U=None)
    # kalman_gain = np.stack([val.item() for val in kf.kalman_gains])
    x = pd.DataFrame(states)
    return x
def kalmanFilterDataframe(x):
    for i, column in enumerate(x.columns):
        states = kalman(x[column])
        x[column] = pd.DataFrame(states)
    return x

def kalmanFilterSeries(x):
    states = kalman(x)
    x = pd.DataFrame(states)
    return x

def simpleFilterDataframe(x, error = 0.02):
    for i, column in enumerate(x.columns):
        max = x[column].max()
        max = abs(max)
        new_array = fft_denoiser(x[column], max * error, True)
        x[column] = pd.DataFrame(new_array)
    return x

def simpleFilterSerie(x, error = 0.02):
    max = x.max()
    max = abs(max)
    new_array = fft_denoiser(x, max * error, True)
    x = pd.DataFrame(new_array)
    return x

def wienerFilterDataframe(x, window=20):
    from scipy.signal import wiener
    for i, column in enumerate(x.columns):
        new_array = wiener(x[column], (window))
        x[column] = pd.DataFrame(new_array)
    return x

def wienerFilterSeries(x, window=20):
    from scipy.signal import wiener
    new_array = wiener(x,(window))
    x = pd.DataFrame(new_array)
    return x

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
def scalingDataFull(training, testing, y_training, y_testing, scalingAlgorithm = 'Standard', scalingOut = None):
    # Scale the 3 dataset
    if scalingAlgorithm == 'None':
        return training, testing, y_training,y_testing, None
    elif scalingAlgorithm == 'Standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scalerOut = StandardScaler()
    elif scalingAlgorithm == 'Normalizer':
        from sklearn.preprocessing import Normalizer
        scaler = Normalizer()
        scalerOut = Normalizer()
    elif scalingAlgorithm == 'Robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        scalerOut = RobustScaler()
    elif scalingAlgorithm == 'MaxAbsScaler':
        from sklearn.preprocessing import MaxAbsScaler
        scaler = MaxAbsScaler()
        scalerOut = MaxAbsScaler()
    elif scalingAlgorithm == 'Quantile':
        from sklearn.preprocessing import QuantileTransformer
        scaler = QuantileTransformer(n_quantiles=600, output_distribution='normal')
        scalerOut = QuantileTransformer(n_quantiles=600, output_distribution='normal')()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scalerOut = MinMaxScaler(feature_range=(-1, 1))

    scaler.fit(training)
    X_train = scaler.transform(training)
    X_test = scaler.transform(testing)

    if scalingOut != None:
        scalerOut.fit(y_training)
        y_training = scalerOut.transform(y_training)
        y_testing = scalerOut.transform(y_testing)
    else:
        scalerOut = None

    return X_train, X_test, y_training, y_testing, scalerOut

def scalingData(training, testing, scalingAlgorithm = 'Standard'):
    # Scale the 3 dataset
    if scalingAlgorithm == 'Standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif scalingAlgorithm == 'Normalizer':
        from sklearn.preprocessing import Normalizer
        scaler = Normalizer()
    elif scalingAlgorithm == 'Robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    elif scalingAlgorithm == 'MaxAbsScaler':
        from sklearn.preprocessing import MaxAbsScaler
        scaler = MaxAbsScaler()
    elif scalingAlgorithm == 'Quantile':
        from sklearn.preprocessing import QuantileTransformer
        scaler = QuantileTransformer(n_quantiles=600, output_distribution='normal')
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))

    X_train = scaler.fit_transform(training)
    X_test = scaler.fit_transform(testing)

    return X_train, X_test

def clueanUpData(reload = False, filter = None, bestFeature = 0, labelName ='Pitch', printLabel = False, filterLabel = False, winsorizeFactor = 0, trim = False):
    import DataRetrive as es
    data = es.retriveDataSet(reload)

    if (data is None):
        data = es.getDataSet()
        es.saveDataSet(data)
    if (data.empty):
        print('Is the DataFrame empty!')
        raise SystemExit

    # ['Datetime','SinkMin_AP','YawRate','Bs','Heel','Pitch', 'Lwy', 'Tws']
    data = data.drop(['Datetime', 'Bs', 'YawRate', 'Heel', 'Lwy', 'Tws', 'SinkMin_AP'], axis=1)

    if trim==True:
        labelMean = data[labelName].mean()
        data.loc[data['armDown'] == 1, labelName] = labelMean

    data.dropna(inplace=True)
    data.drop(data.tail(5000).index, inplace=True)

    # if parameters['sampling'] > 1:
    #     data = data.rolling(parameters['sampling']).mean()
    #     data = data.iloc[::parameters['sampling'], :]
    #     data.dropna(inplace=True)

    label = data[labelName]
    label_orig = label.copy()
    label_orig = label_orig.values
    label_orig = label_orig.reshape((len(label), 1))

    # get the days of sailing
    zi = data['Day'].values
    # get the index of the last racing day to be used as TestSet
    itemindex = np.where(zi == 6)
    test_index = itemindex[0][0]
    test_index += 25000 #25000

    # remove from the DataFrame the colum of the label
    clean_data = data.drop(labelName, axis=1)

    if filter == None or filter=='none':
        print("No Filtering on data")
    else:
        print("Filtering with ", filter)

        isPort_train = clean_data['isPort']
        isStbd_train = clean_data['isStbd']
        armDown_train = clean_data['armDown']
        day_train = clean_data['Day']
        clean_data = clean_data.drop(['isPort', 'isStbd', 'armDown', 'Day'], axis=1)

        if filter == 'simple':
            clean_data = simpleFilterDataframe(clean_data)
            if filterLabel:
                label = simpleFilterSerie(label)
        elif filter == 'kalman':
            clean_data = kalmanFilterDataframe(clean_data)
            if filterLabel:
                label = kalmanFilterSeries(label)
        else:
            clean_data = wienerFilterDataframe(clean_data)
            if filterLabel:
                label = wienerFilterSeries(label)

        clean_data["isStbd"] = isStbd_train
        clean_data["armDown"] = armDown_train
        clean_data["isPort"] = isPort_train
        clean_data["Day"] = day_train

    if(bestFeature>0):
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_regression
        selector = SelectKBest(score_func=f_regression, k=bestFeature)
        clean_data = selector.fit_transform(clean_data, label)
        Xi = clean_data
        yi = label
    else:
        Xi = clean_data.values

    yi = label.values
    yi = yi.reshape((len(yi), 1))

    TEST_SPLIT = 1 - (test_index / len(Xi))  # Test Split to last racing day

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(Xi, yi, test_size=TEST_SPLIT, shuffle=False)

    if winsorizeFactor > 0:
        from scipy.stats.mstats import winsorize
        winsorize(X_train, limits=[winsorizeFactor, winsorizeFactor])
        winsorize(X_test, limits=[winsorizeFactor, winsorizeFactor])
        winsorize(y_train, limits=[winsorizeFactor, winsorizeFactor])
        winsorize(y_test, limits=[winsorizeFactor, winsorizeFactor])

    if (printLabel):
        import matplotlib.pyplot as pltDataSet
        range_history = len(yi)
        range_future = len(y_test)
        range_forecast = list(range(test_index, range_history))
        range_full = len(label_orig)
        pltDataSet.figure(figsize=(10, 5))
        pltDataSet.plot(np.arange(range_history), np.array(yi), label=labelName)
        pltDataSet.plot(np.arange(len(data['Day'].values)), data['Day'].values, label='Days')
        pltDataSet.plot(range_forecast, np.array(y_test), label='TestSet')
        pltDataSet.plot(np.arange(range_full), np.array(label_orig), label='Real', alpha=0.5)
        pltDataSet.title("Full DataSet")
        pltDataSet.xlabel('Time step', fontsize=18)
        pltDataSet.legend(loc='upper right')
        pltDataSet.ylabel('Values', fontsize=18)
        pltDataSet.legend()
        pltDataSet.ion()
        pltDataSet.draw()
        pltDataSet.pause(0.1)
        pltDataSet.show(block=True)

    del (label_orig)
    del(clean_data)
    del(Xi)
    del(yi)

    return X_train, X_test, y_train, y_test


def toSplitSequence(X_train, X_test, y_train, y_test, n_timestamps = 20, n_future = 1):
    dataset = np.append(X_train, y_train, axis=1)
    X_train, y_train = split_sequences(dataset, n_timestamps, n_future)
    dataset = np.append(X_test, y_test, axis=1)
    X_test, y_test = split_sequences(dataset, n_timestamps, n_future)
    del(dataset)
    return X_train, X_test, y_train, y_test

def model_performance(model, X, y):
    """
    Get accuracy score on validation/test data from a trained model
    """
    DEFAULT_RETURN = 0.4

    try:
        y_pred = model.predict(X)
        y_test = y[:, 0]
        r2 = metrics.r2_score(y_test, y_pred)
        print("R2 Testing: ", r2)
        return round(r2, 3)
    except:
        print("R2 Testing FAILS")
        return DEFAULT_RETURN

def get_optimizer(optimizer = 'adam', learning_rate = '0.001', decay = None, decay_steps = 4500, decay_rate = 0.5):

    if decay == 'time':
        lr = tf.keras.optimizers.schedules.InverseTimeDecay(
            learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=False)
    elif decay == 'expo':
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate, #0.96
            staircase=False)
    else:
        if optimizer == 'adam':
            if learning_rate == 0:
                lr = 0.001  # default
            else:
                lr = learning_rate
        else:
            if learning_rate == 0:
                lr = 0.01  # default
            else:
                lr = learning_rate

    if optimizer == 'adam':
        opti = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        momentum = 0.9 #default
        nesterov = True
        opti = tf.keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=momentum, nesterov=nesterov)
    return opti

def get_lr_metric(optimizer):
    lr = optimizer.learning_rate
    if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
        return lr(optimizer.iterations)
    else:
        return lr

def get_activation(activation = "tanh"):
    if activation == "relu":
        act_func = relu
    elif activation == "elu":
        act_func = elu
    elif activation == "sigmoid":
        act_func = sigmoid
    else:
        act_func = tanh
    return act_func

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

def LearningRatePlot(model, X_train,y_train, epochs = 2, batch_size = 64):
    # model is a Keras model
    import LRFinderKeras
    lr_finder = LRFinderKeras.LRFinder(model)
    # Train a model with batch size 512 for 5 epochs
    # with learning rate growing exponentially from 0.0001 to 1
    lr_finder.find(X_train, y_train, start_lr=1e-7, end_lr=10, batch_size=batch_size, epochs=epochs)
    lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)
    lr_finder.plot_loss_change(sma=20, n_skip_beginning=20, n_skip_end=5, y_lim=(-0.005, 0.001))