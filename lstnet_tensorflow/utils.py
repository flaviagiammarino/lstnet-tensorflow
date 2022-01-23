import numpy as np

def get_training_sequences(y, n_lookback):

    '''
    Split the time series into input sequences and output values. These are used for training the model.
    See Sections 3.1 and 3.8 in the LSTNet paper.

    Parameters:
    __________________________________
    y: np.array.
        Time series, array with shape (n_samples, n_targets) where n_samples is the length of the time
        series and n_targets is the number of time series.

    n_lookback: int.
        The number of past time steps used as input.

    Returns:
    __________________________________
    X: np.array.
        Input sequences, array with shape (n_samples - n_lookback, n_lookback, n_targets).

    Y: np.array.
        Output values, array with shape (n_samples - n_lookback, n_targets).
    '''

    X = np.zeros((y.shape[0], n_lookback, y.shape[1]))
    Y = np.zeros((y.shape[0], y.shape[1]))

    for i in range(n_lookback, y.shape[0]):

        X[i, :, :] = y[i - n_lookback: i, :]
        Y[i, :] = y[i, :]

    X = X[n_lookback:, :, :]
    Y = Y[n_lookback:, :]

    return X, Y
