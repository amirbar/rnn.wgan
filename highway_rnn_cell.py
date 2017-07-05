import tensorflow as tf
from multiplicative_integration import multiplicative_integration
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import RNNCell


def ntanh(_x, name="normalizing_tanh"):
    """
    Inspired by self normalizing networks paper, we adjust scale on tanh
    function to encourage mean of 0 and variance of 1 in activations

    From comments on reddit, the normalizing tanh function:
    1.5925374197228312
    """
    scale = 1.5925374197228312
    return scale*tf.nn.tanh(_x, name=name)


class RHNCell(RNNCell):
  """
  Recurrent Highway Cell
  Reference: https://arxiv.org/abs/1607.03474
  """

  def __init__(self, num_units, depth=2, forget_bias=-2.0, activation=ntanh):

    """We initialize forget bias to negative two so that highway layers don't activate
    """
    

    assert activation.__name__ == "ntanh"
    self._num_units = num_units
    self._in_size = num_units
    self._depth = depth
    self._forget_bias = forget_bias
    self._activation = activation

    tf.logging.info("""Building Recurrent Highway Cell with {} Activation of depth {} 
      and forget bias of {}""".format(
        self._activation.__name__, self._depth, self._forget_bias))
    

  @property
  def input_size(self):
    return self._in_size

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, timestep=0, scope=None):
    current_state = state

    for i in range(self._depth):
      with tf.variable_scope('h_'+str(i)):
        if i == 0:
          h = self._activation(
            multiplicative_integration([inputs,current_state], self._num_units))
        else:
          h = tf.layers.dense(current_state, self._num_units, self._activation, 
                bias_initializer=tf.zeros_initializer())

      with tf.variable_scope('gate_'+str(i)):
        if i == 0:
          t = tf.sigmoid(
            multiplicative_integration([inputs,current_state], self._num_units, 
              self._forget_bias))

        else:
          t = tf.layers.dense(current_state, self._num_units, tf.sigmoid, 
                bias_initializer=tf.constant_initializer(self._forget_bias))

      current_state = (h - current_state)* t + current_state

    return current_state, current_state