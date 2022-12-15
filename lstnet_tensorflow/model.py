import pandas as pd
import numpy as np
import tensorflow as tf
pd.options.mode.chained_assignment = None

from lstnet_tensorflow.layers import SkipGRU
from lstnet_tensorflow.utils import get_training_sequences

class LSTNet():

    def __init__(self,
                 y,
                 forecast_period,
                 lookback_period,
                 filters=100,
                 kernel_size=3,
                 gru_units=100,
                 skip_gru_units=50,
                 skip=1,
                 lags=1,
                 dropout=0,
                 regularizer='L2',
                 regularization_factor=0.01):

        '''
        Implementation of multivariate time series forecasting model introduced in Lai, G., Chang, W. C., Yang, Y.,
        & Liu, H. (2018). Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks. In "The 41st
        International ACM SIGIR Conference on Research & Development in Information Retrieval" (SIGIR '18).
        Association for Computing Machinery, New York, NY, USA, 95–104. 

        Parameters:
        __________________________________
        y: np.array.
            Time series, array with shape (n_samples, n_targets) where n_samples is the length of the time series
            and n_targets is the number of time series.

        forecast_period: int.
            Number of future time steps to forecast.

        lookback_period: int.
            Number of past time steps to use as input.

        filters: int.
            Number of filters (or channels) of the convolutional layer.

        kernel_size: int.
            Kernel size of the convolutional layer.

        gru_units: list.
            Hidden units of GRU layer.

        skip_gru_units: list.
            Hidden units of Skip GRU layer.

        skip: int.
            Number of skipped hidden cells in the Skip GRU layer.

        lags: int.
            Number of autoregressive lags.

        dropout: float.
            Dropout rate.

        regularizer: str.
            Regularizer, either 'L1', 'L2' or 'L1L2'.

        regularization_factor: float.
            Regularization factor.
        '''

        # Normalize the targets.
        y_min, y_max = np.min(y, axis=0), np.max(y, axis=0)
        y = (y - y_min) / (y_max - y_min)
        self.y_min = y_min
        self.y_max = y_max

        # Extract the input sequences and output values.
        self.X, self.Y = get_training_sequences(y, lookback_period)

        # Save the inputs.
        self.y = y
        self.n_samples = y.shape[0]
        self.n_targets = y.shape[1]
        self.n_lookback = lookback_period
        self.n_forecast = forecast_period

        # Build and save the model.
        self.model = build_fn(
            self.n_targets,
            self.n_lookback,
            filters,
            kernel_size,
            gru_units,
            skip_gru_units,
            skip,
            lags,
            dropout,
            regularizer,
            regularization_factor)

    def fit(self,
            loss='mse',
            learning_rate=0.001,
            batch_size=32,
            epochs=100,
            validation_split=0,
            verbose=1):

        '''
        Train the model.

        Parameters:
        __________________________________
        loss: str, function.
            Loss function, see https://www.tensorflow.org/api_docs/python/tf/keras/losses.

        learning_rate: float.
            Learning rate.

        batch_size: int.
            Batch size.

        epochs: int.
            Number of epochs.

        validation_split: float.
            Fraction of the training data to be used as validation data, must be between 0 and 1.

        verbose: int.
            Verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch.
        '''

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
        )

        self.model.fit(
            x=self.X,
            y=self.Y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )

    def forecast(self, y):
    
        '''
        Generate the forecasts.
        
        Parameters:
        __________________________________
        y: np.array.
            Past values of the time series.

        Returns:
        __________________________________
        forecasts: pd.DataFrame.
            Data frame including the actual values of the time series and the forecasts.
        '''

        # Normalize the targets.
        y = (y - self.y_min) / (self.y_max - self.y_min)
        
        # Generate the multi-step forecasts.
        x_pred = y[- self.n_lookback - 1: - 1, :].reshape(1, self.n_lookback, self.n_targets)   # Last observed input sequence.
        y_pred = y[-1:, :].reshape(1, 1, self.n_targets)                                        # Last observed target value.
        y_future = []                                                                           # Future target values.
        
        for i in range(self.n_forecast):

            # Feed the last forecast back to the model as an input.
            x_pred = np.append(x_pred[:, 1:, :], y_pred, axis=1)

            # Generate the next forecast.
            y_pred = self.model(x_pred).numpy().reshape(1, 1, self.n_targets)

            # Save the forecast.
            y_future.append(y_pred.flatten().tolist())

        y_future = np.array(y_future)

        # Organize the forecasts in a data frame.
        columns = ['time_idx']
        columns.extend(['actual_' + str(i + 1) for i in range(self.n_targets)])
        columns.extend(['predicted_' + str(i + 1) for i in range(self.n_targets)])

        df = pd.DataFrame(columns=columns)
        df['time_idx'] = np.arange(self.n_samples + self.n_forecast)

        for i in range(self.n_targets):

            df['actual_' + str(i + 1)].iloc[: - self.n_forecast] = \
                self.y_min[i] + (self.y_max[i] - self.y_min[i]) * self.y[:, i]

            df['predicted_' + str(i + 1)].iloc[- self.n_forecast:] = \
                self.y_min[i] + (self.y_max[i] - self.y_min[i]) * y_future[:, i]

        # Return the data frame.
        return df.astype(float)


def build_fn(n_targets,
             n_lookback,
             filters,
             kernel_size,
             gru_units,
             skip_gru_units,
             skip,
             lags,
             dropout,
             regularizer,
             regularization_factor):

    '''
    Build the model, see Section 3 in the LSTNet paper.

    Parameters:
    __________________________________
    n_targets: int.
        Number of time series.

    n_lookback: int.
        Number of past time steps to use as input.

    filters: int.
        Number of filters (or channels) of the convolutional layer.

    kernel_size: int.
        Kernel size of the convolutional layer.

    gru_units: list.
        Hidden units of GRU layer.

    skip_gru_units: list.
        Hidden units of Skip GRU layer.

    skip: int.
        Number of skipped hidden cells in the Skip GRU layer.

    lags: int.
        Number of autoregressive lags.

    dropout: float.
        Dropout rate.

    regularizer: str.
        Regularizer, either 'L1', 'L2' or 'L1L2'.

    regularization_factor: float.
        Regularization factor.
    '''

    # Inputs.
    x = tf.keras.layers.Input(shape=(n_lookback, n_targets))

    # Convolutional component, see Section 3.2 in the LSTNet paper.
    c = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(x)
    c = tf.keras.layers.Dropout(rate=dropout)(c)

    # Recurrent component, see Section 3.3 in the LSTNet paper.
    r = tf.keras.layers.GRU(units=gru_units, activation='relu')(c)
    r = tf.keras.layers.Dropout(rate=dropout)(r)

    # Recurrent-skip component, see Section 3.4 in the LSTNet paper.
    s = SkipGRU(units=skip_gru_units, activation='relu', return_sequences=True)(c)
    s = tf.keras.layers.Dropout(rate=dropout)(s)
    s = tf.keras.layers.Lambda(function=lambda x: x[:, - skip:, :])(s)
    s = tf.keras.layers.Reshape(target_shape=(s.shape[1] * s.shape[2],))(s)
    d = tf.keras.layers.Concatenate(axis=1)([r, s])
    d = tf.keras.layers.Dense(units=n_targets, kernel_regularizer=kernel_regularizer(regularizer, regularization_factor))(d)

    # Autoregressive component, see Section 3.6 in the LSTNet paper.
    l = tf.keras.layers.Flatten()(x[:, - lags:, :])
    l = tf.keras.layers.Dense(units=n_targets, kernel_regularizer=kernel_regularizer(regularizer, regularization_factor))(l)

    # Outputs.
    y = tf.keras.layers.Add()([d, l])

    return tf.keras.models.Model(x, y)


def kernel_regularizer(regularizer, regularization_factor):

    '''
    Define the kernel regularizer.

    Parameters:
    __________________________________
    regularizer: str.
        Regularizer, either 'L1', 'L2' or 'L1L2'.

    regularization_factor: float.
        Regularization factor.
    '''

    if regularizer == 'L1':
        return tf.keras.regularizers.L1(l1=regularization_factor)

    elif regularizer == 'L2':
        return tf.keras.regularizers.L2(l2=regularization_factor)

    elif regularizer == 'L1L2':
        return tf.keras.regularizers.L1L2(l1=regularization_factor, l2=regularization_factor)

    else:
        raise ValueError('Undefined regularizer {}.'.format(regularizer))
