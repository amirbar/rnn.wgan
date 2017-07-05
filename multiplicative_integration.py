import tensorflow as tf

def multiplicative_integration(list_of_inputs, output_size, initial_bias_value=0.0,
  weights_already_calculated=False, scope=None):
    """Multiplicative Integration from https://arxiv.org/abs/1606.06630

    expects len(2) for list of inputs and will perform integrative multiplication

    weights_already_calculated will treat the list of inputs as Wx and Uz and is useful for batch normed inputs
    """
    with tf.variable_scope(scope or 'double_inputs_multiple_integration'):
      if len(list_of_inputs) != 2: 
        raise ValueError('list of inputs must be 2, you have: {}'.format(len(list_of_inputs)))
      
      if weights_already_calculated:
        Wx = list_of_inputs[0]
        Uz = list_of_inputs[1]

      else:
        with tf.variable_scope('Calculate_Wx_mulint'):
          Wx = tf.layers.dense(list_of_inputs[0], output_size, use_bias=False)

        with tf.variable_scope("Calculate_Uz_mulint"):
          Uz = tf.layers.dense(list_of_inputs[1], output_size, use_bias=False)

      with tf.variable_scope("multiplicative_integration"):
        alpha = tf.get_variable('mulint_alpha', [output_size], 
            initializer = tf.truncated_normal_initializer(mean=1.0, stddev=0.1))

        # For efficiency, we retrieve both beta parameters via tf split
        beta1, beta2 = tf.split(
          tf.get_variable('mulint_params_betas', [output_size*2], 
            initializer = tf.truncated_normal_initializer(mean=0.5, stddev=0.1)),
          num_or_size_splits=2,
          axis=0) 

        original_bias = tf.get_variable('mulint_original_bias', [output_size], 
            initializer = tf.truncated_normal_initializer(mean=initial_bias_value, stddev=0.1)) 

      final_output = alpha*Wx*Uz + beta1*Uz + beta2*Wx + original_bias

    return final_output