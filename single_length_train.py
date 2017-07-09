import sys

from tensorflow.python.training.saver import latest_checkpoint

from config import *
from language_helpers import generate_argmax_samples_and_gt_samples, inf_train_gen, decode_indices_to_string
from objective import get_optimization_ops, define_objective
from summaries import define_summaries, \
    log_samples
import numpy as np

sys.path.append(os.getcwd())

from model import *
import model_and_data_serialization


# Download Google Billion Word at http://www.statmt.org/lm-benchmark/ and
# fill in the path to the extracted files here!

def run(iterations, seq_length, is_first, charmap, inv_charmap, prev_seq_length):
    if len(DATA_DIR) == 0:
        raise Exception('Please specify path to data directory in single_length_train.py!')

    lines, _, _ = model_and_data_serialization.load_dataset(seq_length=seq_length, b_charmap=False, b_inv_charmap=False, n_examples=FLAGS.MAX_N_EXAMPLES)

    real_inputs_discrete = tf.placeholder(tf.int32, shape=[BATCH_SIZE, seq_length])

    global_step = tf.Variable(0, trainable=False)
    disc_cost, gen_cost, fake_inputs, disc_fake, disc_real, disc_on_inference, inference_op, other_ops = define_objective(charmap,real_inputs_discrete, seq_length,
        gan_type=FLAGS.GAN_TYPE, rnn_cell=RNN_CELL)


    merged, train_writer = define_summaries(disc_cost, gen_cost, seq_length)
    disc_train_op, gen_train_op = get_optimization_ops(
        disc_cost, gen_cost, global_step, FLAGS.DISC_LR, FLAGS.GEN_LR)

    saver = tf.train.Saver(tf.trainable_variables())

    # Use JIT XLA compilation to speed up calculations
    config=tf.ConfigProto(
        log_device_placement=False, allow_soft_placement=True)
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    with tf.Session(config=config) as session:

        session.run(tf.initialize_all_variables())
        if not is_first:
            print("Loading previous checkpoint...")
            internal_checkpoint_dir = model_and_data_serialization.get_internal_checkpoint_dir(prev_seq_length)
            model_and_data_serialization.optimistic_restore(session,
                latest_checkpoint(internal_checkpoint_dir, "checkpoint"))

            restore_config.set_restore_dir(
                load_from_curr_session=True)  # global param, always load from curr session after finishing the first seq

        gen = inf_train_gen(lines, charmap)
        _gen_cost_list = []
        _disc_cost_list = []
        _step_time_list = []

        for iteration in range(iterations):
            start_time = time.time()

            # Train critic
            for i in range(CRITIC_ITERS):
                _data = next(gen)

                if FLAGS.GAN_TYPE.lower() == "fgan":
                    _disc_cost, _, real_scores, _ = session.run(
                    [disc_cost, disc_train_op, disc_real,
                        other_ops["alpha_optimizer_op"]],
                    feed_dict={real_inputs_discrete: _data}
                    )

                elif FLAGS.GAN_TYPE.lower() == "wgan":
                    _disc_cost, _, real_scores = session.run(
                    [disc_cost, disc_train_op, disc_real],
                    feed_dict={real_inputs_discrete: _data}
                    )

                else:
                    raise ValueError(
                        "Appropriate gan type not selected: {}".format(FLAGS.GAN_TYPE))
                _disc_cost_list.append(_disc_cost)



            # Train G
            for i in range(GEN_ITERS):
                _data = next(gen)
                # in Fisher GAN, paper measures convergence by gen_cost instead of disc_cost
                # To measure convergence, gen_cost should start at a positive number and decrease
                # to zero. The lower, the better.
                _gen_cost, _ = session.run([gen_cost, gen_train_op], feed_dict={real_inputs_discrete: _data})
                _gen_cost_list.append(_gen_cost)

            _step_time_list.append(time.time() - start_time)

            if FLAGS.PRINT_EVERY_STEP:
                print("iteration %s/%s"%(iteration, iterations))
                print("disc cost {}"%_disc_cost)
                print("gen cost {}".format(_gen_cost))
                print("total step time {}".format(time.time() - start_time))


            # Summaries
            if iteration % FLAGS.PRINT_ITERATION == FLAGS.PRINT_ITERATION-1:
                _data = next(gen)
                summary_str = session.run(
                    merged,
                    feed_dict={real_inputs_discrete: _data}
                )

                tf.logging.warn("iteration %s/%s"%(iteration, iterations))
                tf.logging.warn("disc cost {} gen cost {} average step time {}".format(
                    np.mean(_disc_cost_list), np.mean(_gen_cost_list), np.mean(_step_time_list)))
                _gen_cost_list, _disc_cost_list, _step_time_list = [], [], []

                train_writer.add_summary(summary_str, global_step=iteration)
                fake_samples, samples_real_probabilites, fake_scores = generate_argmax_samples_and_gt_samples(session, inv_charmap, fake_inputs, disc_fake, gen, real_inputs_discrete,feed_gt=True)

                log_samples(fake_samples, fake_scores, iteration, seq_length, "train")
                log_samples(decode_indices_to_string(_data, inv_charmap), real_scores, iteration, seq_length,
                             "gt")
                test_samples, _, fake_scores = generate_argmax_samples_and_gt_samples(session,
                  inv_charmap,
                  inference_op,
                  disc_on_inference,
                  gen,
                  real_inputs_discrete,
                  feed_gt=False)
                # disc_on_inference, inference_op
                log_samples(test_samples, fake_scores, iteration, seq_length, "test")


            if iteration % FLAGS.SAVE_CHECKPOINTS_EVERY == FLAGS.SAVE_CHECKPOINTS_EVERY-1:
                saver.save(session, model_and_data_serialization.get_internal_checkpoint_dir(seq_length) + "/ckp")

        saver.save(session, model_and_data_serialization.get_internal_checkpoint_dir(seq_length) + "/ckp")
        session.close()
