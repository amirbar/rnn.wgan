import tensorflow as tf
from config import *


def Discriminator_RNN(inputs, charmap_len, seq_len, reuse=False, rnn_cell=None):
    with tf.variable_scope("Discriminator", reuse=reuse):
        num_neurons = FLAGS.DISC_STATE_SIZE

        weight = tf.get_variable("embedding", shape=[charmap_len, num_neurons],
            initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

        # backwards compatibility
        if FLAGS.DISC_RNN_LAYERS == 1:
            cell = rnn_cell(num_neurons)
        else:
            cell = tf.contrib.rnn.MultiRNNCell([rnn_cell(num_neurons) for _ in range(FLAGS.DISC_RNN_LAYERS)], state_is_tuple=True)

        flat_inputs = tf.reshape(inputs, [-1, charmap_len])

        inputs = tf.reshape(tf.matmul(flat_inputs, weight), [-1, seq_len, num_neurons])
        inputs = tf.unstack(tf.transpose(inputs, [1,0,2]))


        for inp in inputs:
            print(inp.get_shape())

        output, state = tf.contrib.rnn.static_rnn(
            cell,
            inputs,
            dtype=tf.float32
        )

        last = output[-1]

        weight = tf.get_variable("W", shape=[num_neurons, 1],
                                 initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        bias = tf.get_variable("b", shape=[1], initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

        prediction = tf.matmul(last, weight) + bias

        return prediction


def Generator_RNN_CL_VL_TH(n_samples, charmap_len, seq_len=None, gt=None, rnn_cell=None):
    with tf.variable_scope("Generator"):
        noise, noise_shape = get_noise()
        num_neurons = FLAGS.GEN_STATE_SIZE

        cells = []
        for l in range(FLAGS.GEN_RNN_LAYERS):
            cells.append(rnn_cell(num_neurons))

        # this is separate to decouple train and test
        train_initial_states = create_initial_states(noise)
        inference_initial_states = create_initial_states(noise)

        sm_weight = tf.Variable(tf.random_uniform([num_neurons, charmap_len], minval=-0.1, maxval=0.1))
        sm_bias = tf.Variable(tf.random_uniform([charmap_len], minval=-0.1, maxval=0.1))

        embedding = tf.Variable(tf.random_uniform([charmap_len, num_neurons], minval=-0.1, maxval=0.1))

        char_input = tf.Variable(tf.random_uniform([num_neurons], minval=-0.1, maxval=0.1))
        char_input = tf.reshape(tf.tile(char_input, [n_samples]), [n_samples, 1, num_neurons])

        if seq_len is None:
            seq_len = tf.placeholder(tf.int32, None, name="ground_truth_sequence_length")

        if gt is not None: #if no GT, we are training
            train_pred = get_train_op(cells, char_input, charmap_len, embedding, gt, n_samples, num_neurons, seq_len, sm_bias, sm_weight, train_initial_states)
            inference_op = get_inference_op(cells, char_input, embedding, seq_len, sm_bias, sm_weight, inference_initial_states,
                num_neurons,
                charmap_len, reuse=True)
        else:
            inference_op = get_inference_op(cells, char_input, embedding, seq_len, sm_bias, sm_weight, inference_initial_states,
                num_neurons,
                charmap_len, reuse=False)
            train_pred = None

        return train_pred, inference_op


def create_initial_states(noise):
    states = []
    for l in range(FLAGS.GEN_RNN_LAYERS):
        states.append(noise)
    return states


def get_train_op(cells, char_input, charmap_len, embedding, gt, n_samples, num_neurons, seq_len, sm_bias, sm_weight, states):
    gt_embedding = tf.reshape(gt, [n_samples * seq_len, charmap_len])
    gt_RNN_input = tf.matmul(gt_embedding, embedding)
    gt_RNN_input = tf.reshape(gt_RNN_input, [n_samples, seq_len, num_neurons])[:, :-1]
    gt_sentence_input = tf.concat([char_input, gt_RNN_input], axis=1)
    RNN_output, _ = rnn_step_prediction(cells, charmap_len, gt_sentence_input, num_neurons, seq_len, sm_bias, sm_weight, states)
    train_pred = []
    # TODO: optimize loop
    for i in range(seq_len):
        train_pred.append(
            tf.concat([tf.zeros([BATCH_SIZE, seq_len - i - 1, charmap_len]), gt[:, :i], RNN_output[:, i:i + 1, :]],
                      axis=1))

    train_pred = tf.reshape(train_pred, [BATCH_SIZE*seq_len, seq_len, charmap_len])

    if FLAGS.LIMIT_BATCH:
        indices = tf.random_uniform([BATCH_SIZE], 0, BATCH_SIZE*seq_len, dtype=tf.int32)
        train_pred = tf.gather(train_pred, indices)

    return train_pred


def rnn_step_prediction(cells, charmap_len, gt_sentence_input, num_neurons, seq_len, sm_bias, sm_weight, states,
                        reuse=False):
    with tf.variable_scope("rnn", reuse=reuse):
        RNN_output = gt_sentence_input
        for l in range(FLAGS.GEN_RNN_LAYERS):
            RNN_output, states[l] = tf.nn.dynamic_rnn(cells[l], RNN_output, dtype=tf.float32,
               initial_state=states[l], scope="layer_%d" % (l + 1))
    RNN_output = tf.reshape(RNN_output, [-1, num_neurons])
    RNN_output = tf.nn.softmax(tf.matmul(RNN_output, sm_weight) + sm_bias)
    RNN_output = tf.reshape(RNN_output, [BATCH_SIZE, -1, charmap_len])
    return RNN_output, states


def get_inference_op(cells, char_input, embedding, seq_len, sm_bias, sm_weight, states, num_neurons, charmap_len,
                     reuse=False):
    inference_pred = []
    embedded_pred = [char_input]
    for i in range(seq_len):
        step_pred, states = rnn_step_prediction(cells, charmap_len, tf.concat(embedded_pred, 1), num_neurons, seq_len,
                                                sm_bias, sm_weight, states, reuse=reuse)
        best_chars_tensor = tf.argmax(step_pred, axis=2)
        best_chars_one_hot_tensor = tf.one_hot(best_chars_tensor, charmap_len)
        best_char = best_chars_one_hot_tensor[:, -1, :]
        inference_pred.append(tf.expand_dims(best_char, 1))
        embedded_pred.append(tf.expand_dims(tf.matmul(best_char, embedding), 1))
        reuse = True  # no matter what the reuse was, after the first step we have to reuse the defined vars

    return tf.concat(inference_pred, axis=1)


generators = {
    "Generator_RNN_CL_VL_TH": Generator_RNN_CL_VL_TH,
}

discriminators = {
    "Discriminator_RNN": Discriminator_RNN,
}

def get_noise():
    noise_shape = [BATCH_SIZE, FLAGS.GEN_STATE_SIZE]
    return make_noise(shape=noise_shape, stddev=FLAGS.NOISE_STDEV), noise_shape


def get_generator(model_name):
    return generators[model_name]


def params_with_name(name):
    return [p for p in tf.trainable_variables() if name in p.name]


def get_discriminator(model_name):
    return discriminators[model_name]


def make_noise(shape, mean=0.0, stddev=1.0):
    return tf.random_normal(shape, mean, stddev)