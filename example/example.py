import numpy as np

from lstnet_tensorflow.model import LSTNet
from lstnet_tensorflow.plots import plot

# Generate some time series
N = 500
t = np.linspace(0, 1, N)
e = np.random.multivariate_normal(mean=np.zeros(3), cov=np.eye(3), size=N)
a = 10 + 10 * t + 10 * np.cos(2 * np.pi * (10 * t - 0.5)) + 1 * e[:, 0]
b = 20 + 20 * t + 20 * np.cos(2 * np.pi * (20 * t - 0.5)) + 2 * e[:, 1]
c = 30 + 30 * t + 30 * np.cos(2 * np.pi * (30 * t - 0.5)) + 3 * e[:, 2]
y = np.hstack([a.reshape(-1, 1), b.reshape(-1, 1), c.reshape(-1, 1)])

# Fit the model
model = LSTNet(
    y=y,
    forecast_period=100,
    lookback_period=200,
    kernel_size=3,
    filters=4,
    gru_units=4,
    skip_gru_units=3,
    skip=50,
    lags=100,
)

model.fit(
    loss='mse',
    learning_rate=0.01,
    batch_size=64,
    epochs=100,
    verbose=1
)

# Generate the forecasts
df = model.forecast(y=y)

# Plot the forecasts
fig = plot(df=df)
fig.write_image('results.png', scale=4, height=900, width=700)