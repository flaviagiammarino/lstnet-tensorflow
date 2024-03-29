#---------------------------------------------- Set Up ----------------------------------------------#
import os
import random
import numpy as np
import tensorflow as tf

from lstnet_tensorflow.layers import SkipGRU

def reset_random_seeds():
   # https://stackoverflow.com/a/61700482/11989081
   os.environ['PYTHONHASHSEED'] = str(42)
   tf.random.set_seed(42)
   np.random.seed(42)
   random.seed(42)
   
# Parameters.
p = 1
seed = 1
units = 2
timesteps = 4
samples = 6
features = 8
epochs = 10
batch_size = 1

# Data.
np.random.seed(seed)
x_ = np.random.normal(0, 1, (samples, timesteps, features))
y_ = np.random.normal(0, 1, samples)

#---------------------------------------- Test 1: Untrained models ----------------------------------#

# Untrained Skip-GRU.
reset_random_seeds()
i1 = tf.keras.layers.Input(shape=(timesteps, features))
o1, s1 = SkipGRU(units=units, p=p, return_state=True, return_sequences=True, activation='relu')(i1)
m1 = tf.keras.models.Model(i1, [o1, s1])
y1, h1 = m1(x_)

# Untrained GRU.
reset_random_seeds()
i2 = tf.keras.layers.Input(shape=(timesteps, features))
o2, s2 = tf.keras.layers.GRU(units=units, return_state=True, return_sequences=True, activation='relu')(i2)
m2 = tf.keras.models.Model(i2, [o2, s2])
y2, h2 = m2(x_)

# Outputs comparison, should be True when p = 1 and False otherwise.
print(np.isclose(y1.numpy(), y2.numpy()).sum() == np.prod([samples, timesteps, units]))

# States comparison, should be True when p = 1 and False otherwise.
print(np.isclose(h1.numpy(), h2.numpy()).sum() == np.prod([samples, units]))

#---------------------------------------- Test 2: Trained models ----------------------------------#

# Trained Skip-GRU.
reset_random_seeds()
i1 = tf.keras.layers.Input(shape=(timesteps, features))
o1 = SkipGRU(units=units, p=p, activation='relu')(i1)
m1 = tf.keras.models.Model(i1, o1)
m1.compile(loss='mse', optimizer='adam')
hist1 = m1.fit(x_, y_, epochs=epochs, batch_size=batch_size, verbose=0)
pred1 = m1.predict(x_)

# Trained GRU.
reset_random_seeds()
i2 = tf.keras.layers.Input(shape=(timesteps, features))
o2 = tf.keras.layers.GRU(units=units, activation='relu')(i2)
m2 = tf.keras.models.Model(i2, o2)
m2.compile(loss='mse', optimizer='adam')
hist2 = m2.fit(x_, y_, epochs=epochs, batch_size=batch_size, verbose=0)
pred2 = m2.predict(x_)

# Predictions comparison, should be True when p = 1 and False otherwise.
print(np.isclose(pred1, pred2).sum() == np.prod([samples, units]))

# Loss comparison, should be True when p = 1 and False otherwise.
print(np.isclose(hist1.history['loss'], hist2.history['loss']).sum() == epochs)
