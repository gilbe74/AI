import numpy as np
import pandas as pd
from numpy import array
from numpy import minimum

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