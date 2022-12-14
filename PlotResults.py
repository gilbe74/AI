import math
import Utility as ut
import numpy as np
import pandas as pd

import tensorflow.keras as keras

def PlotResult(model, X, y, history, run=None, batch_size = 64, n_future = 1, saveDelta = True, SENSOR_ERROR = 0.05, filter_out = 'none', scaler = None, test_unscaled = None):
    print("Valuate the model")

    SINGLEPOINT = int(len(X) / 2)

    if model == None:
        model = keras.models.load_model('best_model.h5')

    # Evaluate the model on the test data using `evaluate`
    test_results = model.evaluate(X, y, batch_size=batch_size)
    # Log predictions as table
    test_pred = model.predict(X)

    if(scaler != None):
        test_pred = scaler.inverse_transform(test_pred)
        y = scaler.inverse_transform(y)
        # lastElementIndex = len(test_pred) - 1
        # # Removing the last element using slicing
        # test_pred = test_pred[:lastElementIndex]

    test_pred_filtered = pd.DataFrame(test_pred)
    if filter_out != 'none':
        print("Filtering prediction with ", filter_out)
        if filter_out == 'simple':
            test_pred_filtered = ut.simpleFilterSerie(test_pred_filtered)
        elif filter_out == 'kalman':
            test_pred_filtered = ut.kalmanFilterSeries(test_pred)
        else:
            test_pred_filtered = ut.wienerFilterSeries(test_pred)
        test_pred_filtered = test_pred_filtered.to_numpy()
    else:
        print("No Filtering")
        test_pred_filtered = test_pred.copy()

    test_prediction_NON_filtered = test_pred
    test_pred = test_pred_filtered
    del (test_pred_filtered)


    for j, metric in enumerate(test_results):
        if run != None:
            run['test/scores/{}'.format(model.metrics_names[j])] = metric
        print("Metrics {}".format(model.metrics_names[j]), round(metric, 3))

    print("Error Valutation")

    import matplotlib.pyplot as pltError
    error_scores = list()
    for i in range(len(test_pred)):
        real = y[i].item(0)
        expect = test_pred[i].item(0)
        delta = ut.distance(real, expect)
        error_scores.append(delta)
    delta_array = np.array(error_scores)
    if (saveDelta):
        np.save('description_array.npy', delta_array)
    delta_df = pd.DataFrame(delta_array)
    description = delta_df.describe()
    print(delta_df.describe())
    fig = pltError.figure(figsize=(10, 7))
    # pltError.title("Error Description")
    pltError.grid()
    pltError.boxplot(delta_array)
    pltError.draw()
    pltError.pause(0.1)
    if run != None:
        run["evaluation/prediction"].upload(fig)

    range_history = len(y)
    range_future = len(test_pred)
    # start = int(len(test_pred)/2)
    range_forecast = list(range(SINGLEPOINT, SINGLEPOINT + n_future))
    y_test = y[:, 0]
    y_test_pred_single = test_pred[:, 0]

    # R2
    import sklearn.metrics as metrics
    r2 = metrics.r2_score(y_test, test_pred)
    print("R2 Testing: ", r2)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, test_pred))
    print('RMSD ( Root Mean Squared Error ) :', rmse)

    try:
        # Generate predictions (probabilities -- the output of the last layer)
        print("Generate predictions for 1 samples - ", SINGLEPOINT)
        predictions = model.predict(X[SINGLEPOINT:SINGLEPOINT + 1])
        predictions = predictions[:, 0]
        predictions.reshape(-1)
        print('>Expected=%.2f, Predicted=%.2f' % (y_test[SINGLEPOINT:SINGLEPOINT + 1], predictions[0]))
    except:
        print("Error Generate predictions for 1 samples")

    # # Generate predictions (probabilities -- the output of the last layer)
    # print("Generate predictions for 1 samples - ", SINGLEPOINT)
    # predictions = model.predict(X[SINGLEPOINT:SINGLEPOINT + 1])
    # print('>Expected=%.2f, Predicted=%.2f' % (y_test[SINGLEPOINT:SINGLEPOINT + 1], predictions))

    if history!= None:
        import matplotlib.pyplot as pyplotMetrics
        fig, ax = pyplotMetrics.subplots(2, 1)
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
        if run != None:
            run["evaluation/metrics"].upload(fig)

    # # R2
    # import sklearn.metrics as metrics
    # r2 = metrics.r2_score(y_test, test_pred)
    # print("R2 Testing: ", r2)
    # rmse = np.sqrt(metrics.mean_squared_error(y_test, test_pred))
    # print('RMSD ( Root Mean Squared Error ) :', rmse)

    if run != None:
        run["model/parameters/n_epochs"] = len(history.history['loss'])
        run["test/scores/R2"] = r2
        run["test/scores/expected"] = y_test[SINGLEPOINT:SINGLEPOINT + 1]
        run["test/scores/predicted"] = predictions

    test_prediction_NON_filtered = test_prediction_NON_filtered[:, 0]
    test_prediction_NON_filtered = test_prediction_NON_filtered.reshape(-1)

    import matplotlib.pyplot as pltCoere
    # y_test = y[:, 0]
    s1 = np.array(test_pred[:, 0])
    s2 = np.array(y_test)
    s3 = np.array(test_prediction_NON_filtered)
    fig, axs = pltCoere.subplots(2, 1, sharex=True)
    pltCoere.ion()
    axs[0].plot(np.arange(range_history), np.array(y_test), label='Real', color='#1f77b4')  # blue
    axs[0].plot(np.arange(range_future), np.array(test_pred), label='Prediction LSTM', color='#ff7f0e',
                alpha=0.8)
    # axs[0].plot(np.arange(range_future), s3, label='Prediction NoFilter', color='g',
    #             alpha=0.7)
    if n_future > 1:
        axs[0].plot(range_forecast, np.array(test_pred[SINGLEPOINT]), label='Forecasted with LSTM', color='red')
    else:
        axs[0].scatter(SINGLEPOINT, predictions, color="red", marker="x", s=70, label="Single")
    axs[0].legend(loc='upper right')
    # axs[0].set_xlabel('Time step' ,  fontsize=18)
    axs[0].set_ylabel('Pitch', fontsize=10)
    axs[0].legend(loc='upper right')
    axs[0].grid(True)
    # cxy, f = axs[1].cohere(s1, s2, 256, 1. / dt)
    axs[1].xcorr(s1, s2, usevlines=True, maxlags=10, normed=True, lw=2)
    axs[1].set_ylabel('coherence', fontsize=10)
    # adding grid to the graph
    x = np.arange(range_future)
    markerline, stemlines, baseline = pltCoere.stem(x, delta_df, '-', markerfmt=' ')
    pltCoere.setp(baseline, color='r', linewidth=2)
    SENSOR_ERROR = (y_test.max() + abs(y_test.min())) * SENSOR_ERROR / 2
    axs[1].axhspan(SENSOR_ERROR,-SENSOR_ERROR,alpha=.3)
    # axs[1].axhline(SENSOR_ERROR, color="red", linewidth=0.5)
    # axs[1].axhline(-SENSOR_ERROR, color="red", linewidth=0.5)
    axs[1].grid(True)
    axs[1].axhline(0, color='green', lw=2)
    axs[1].set_xlabel('Time step', fontsize=10)
    axs[0].autoscale(axis='y', enable=True)
    axs[1].autoscale(axis='y', enable=True)
    fig.tight_layout()
    pltCoere.draw()
    pltCoere.pause(0.1)
    # s1 = np.array(y_test_pred_single)
    # s2 = np.array(y_test)
    # fig, axs = pltCoere.subplots(2, 1, sharex=True)
    # pltCoere.ion()
    # axs[0].plot(np.arange(range_history), np.array(y_test), label='Real', color='#1f77b4')  # blue
    # axs[0].plot(np.arange(range_future), np.array(y_test_pred_single), label='Prediction LSTM', color='#ff7f0e',
    #             alpha=0.8)
    # if n_future > 1:
    #     axs[0].plot(range_forecast, np.array(test_pred[SINGLEPOINT]), label='Forecasted with LSTM', color='red')
    # else:
    #     axs[0].scatter(SINGLEPOINT, predictions, color="red", marker="x", s=70, label="Single")
    # axs[0].legend(loc='upper right')
    # # axs[0].set_xlabel('Time step' ,  fontsize=18)
    # axs[0].set_ylabel('Pitch', fontsize=10)
    # axs[0].legend(loc='upper right')
    # axs[0].grid(True)
    # # cxy, f = axs[1].cohere(s1, s2, 256, 1. / dt)
    # axs[1].xcorr(s1, s2, usevlines=True, maxlags=10, normed=True, lw=2)
    # axs[1].set_ylabel('coherence', fontsize=10)
    # # adding grid to the graph
    # x = np.arange(range_future)
    # markerline, stemlines, baseline = pltCoere.stem(x, delta_df, '-', markerfmt=' ')
    # pltCoere.setp(baseline, color='r', linewidth=2)
    # axs[1].axhline(SENSOR_ERROR, color="red", linewidth=0.5)
    # axs[1].axhline(-SENSOR_ERROR, color="red", linewidth=0.5)
    # axs[1].grid(True)
    # axs[1].axhline(0, color='green', lw=2)
    # axs[1].set_xlabel('Time step', fontsize=10)
    # axs[0].autoscale(axis='y', enable=True)
    # axs[1].autoscale(axis='y', enable=True)
    # fig.tight_layout()
    # pltCoere.draw()
    # pltCoere.pause(0.1)
    if run != None:
        run["evaluation/prediction"].upload(fig)

    def zoom_in(myplot, delta, gap):  # 3000 = 2.5 minutes
        x_min = SINGLEPOINT - delta
        x_max = SINGLEPOINT + delta
        axs[0].set_xlim([x_min, x_max])
        visible_y_min, visible_y_max = ut.minMax(s2, y_test_pred_single, x_min, x_max, gap)
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
        if run != None:
            run["evaluation/prediction_zoom_" + str(delta)].upload(fig)

    zoom_in(pltCoere, 3000, 0.1)  # 5 minutes
    zoom_in(pltCoere, 200, 0.1)  # 20 seconds

    # Plot Avg
    # axs[0].plot(np.arange(range_history), ut.downsampling(y_test, 10), label='Human', color='yellow', alpha=0.8)
    # axs[0].plot(np.arange(range_future), ut.downsampling(y_test_pred_single, 10), label='Slow', color='pink', alpha=0.8)

    zoom_in(pltCoere, 60, 0.1)  # 6 seconds

    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.ion()
    sns.histplot(delta_df, color="red", label="Errors", kde=True, stat="density", linewidth=3, bins=int(180 / 5))
    plt.title("Error Distribution Label")
    plt.draw()
    # plt.pause(0.1)
    if run != None:
        run["evaluation/error"].upload(fig)

    if run != None:
        run.stop()

    pltCoere.show(block=True)
    pltError.show(block=True)
    plt.show(block=True)

    if history != None:
        pyplotMetrics.show(block=True)
        pyplotMetrics.close()

    pltCoere.close()
    plt.close()
    pltError.close()