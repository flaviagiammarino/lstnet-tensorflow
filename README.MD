# LSTNet TensorFlow

![license](https://img.shields.io/github/license/flaviagiammarino/lstnet-tensorflow)
![languages](https://img.shields.io/github/languages/top/flaviagiammarino/lstnet-tensorflow)

TensorFlow implementation of multivariate time series forecasting model introduced in Lai, G., Chang, W. C., 
Yang, Y., and Liu, H. (2018). Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks. In *The 41st 
International ACM SIGIR Conference on Research & Development in Information Retrieval (SIGIR '18)*. Association for Computing
Machinery, New York, NY, USA, 95–104. [arXiv:1703.07015](https://arxiv.org/abs/1703.07015).

## Dependencies
```bash
pandas==1.3.4
numpy==1.19.5
tensorflow==2.7.0
plotly==5.3.1
kaleido==0.2.1
```
## Usage
```python
import numpy as np
from lstnet_tensorflow.model import LSTNet

# Generate two time series
N = 1000
t = np.linspace(0, 1, N)
e = np.random.multivariate_normal(mean=np.zeros(2), cov=np.eye(2), size=N)
a = 40 + 30 * t + 20 * np.cos(2 * np.pi * (10 * t - 0.5)) + e[:, 0]
b = 50 + 40 * t + 30 * np.cos(2 * np.pi * (20 * t - 0.5)) + e[:, 1]
y = np.hstack([a.reshape(- 1, 1), b.reshape(- 1, 1)])

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
```
```python
# Plot the in-sample predictions
predictions = model.predict(index=900)
fig = model.plot_predictions()
fig.write_image('predictions.png', width=750, height=650)
```
![predictions](example/predictions.png)
```python
# Plot the out-of-sample forecasts
forecasts = model.forecast()
fig = model.plot_forecasts()
fig.write_image('forecasts.png', width=750, height=650)
```
![forecasts](example/forecasts.png)
