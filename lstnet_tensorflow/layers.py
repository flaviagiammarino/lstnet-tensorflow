import tensorflow as tf
from tensorflow.keras.layers import Layer, GRUCell

class SkipGRU(Layer):

    def __init__(self,
                 units,
                 p=1,
                 activation='relu',
                 return_sequences=False,
                 return_state=False,
                 **kwargs):

        '''
        Recurrent-skip layer, see Section 3.4 in the LSTNet paper.
        
        Parameters:
        __________________________________
        units: int.
            Number of hidden units of the GRU cell.

        p: int.
            Number of skipped hidden cells.

        activation: str, function.
            Activation function, see https://www.tensorflow.org/api_docs/python/tf/keras/activations.

        return_sequences: bool.
            Whether to return the last output or the full sequence.

        return_state: bool.
            Whether to return the last state in addition to the output.

        **kwargs: See https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRUCell.
        '''

        if p < 1:
            raise ValueError('The number of skipped hidden cells cannot be less than 1.')

        self.units = units
        self.p = p
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.timesteps = None
        self.cell = GRUCell(units=units, activation=activation, **kwargs)

        super(SkipGRU, self).__init__()

    def build(self, input_shape):

        if self.timesteps is None:
            self.timesteps = input_shape[1]

            if self.p > self.timesteps:
                raise ValueError('The number of skipped hidden cells cannot be greater than the number of timesteps.')

    def call(self, inputs):

        '''
        Parameters:
        __________________________________
        inputs: tf.Tensor.
            Layer inputs, 2-dimensional tensor with shape (n_samples, filters) where n_samples is the batch size
            and filters is the number of channels of the convolutional layer.

        Returns:
        __________________________________
        outputs: tf.Tensor.
            Layer outputs, 2-dimensional tensor with shape (n_samples, units) if return_sequences == False,
            3-dimensional tensor with shape (n_samples, n_lookback, units) if return_sequences == True where
            n_samples is the batch size, n_lookback is the number of past time steps used as input and units
            is the number of hidden units of the GRU cell.

        states: tf.Tensor.
            Hidden states, 2-dimensional tensor with shape (n_samples, units) where n_samples is the batch size
            and units is the number of hidden units of the GRU cell.
        '''

        outputs = tf.TensorArray(
            element_shape=(inputs.shape[0], self.units),
            size=self.timesteps,
            dynamic_size=False,
            dtype=tf.float32
        )

        states = tf.TensorArray(
            element_shape=(inputs.shape[0], self.units),
            size=self.timesteps,
            dynamic_size=False,
            dtype=tf.float32
        )

        initial_states = tf.zeros(
            shape=(tf.shape(inputs)[0], self.units),
            dtype=tf.float32
        )

        for t in tf.range(self.timesteps):

            if t < self.p:
                output, state = self.cell(
                    inputs=inputs[:, t, :],
                    states=initial_states
                )

            else:
                output, state = self.cell(
                    inputs=inputs[:, t, :],
                    states=states.stack()[t - self.p]
                )

            outputs = outputs.write(index=t, value=output)
            states = states.write(index=t, value=state)

        outputs = tf.transpose(outputs.stack(), [1, 0, 2])
        states = tf.transpose(states.stack(), [1, 0, 2])

        if not self.return_sequences:
            outputs = outputs[:, -1, :]

        if self.return_state:
            states = states[:, -1, :]
            return outputs, states

        else:
            return outputs
