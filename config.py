import os
import time

import tensorflow as tf

flags = tf.app.flags




flags.DEFINE_string('LOGS_DIR', './logs/', '')
flags.DEFINE_string('DATA_DIR', './data/1-billion-word-language-modeling-benchmark-r13output/', "")
flags.DEFINE_string('CKPT_PATH', "./ckpt/", "")
flags.DEFINE_integer('BATCH_SIZE', 64, '')
flags.DEFINE_integer('CRITIC_ITERS', 10, '')
flags.DEFINE_integer('LAMBDA', 10, '')
flags.DEFINE_integer('MAX_N_EXAMPLES', 10000000, '')
flags.DEFINE_string('GENERATOR_MODEL', 'Generator_GRU_CL_VL_TH', '')
flags.DEFINE_string('DISCRIMINATOR_MODEL', 'Discriminator_GRU', '')
flags.DEFINE_string('PICKLE_PATH', './pkl', '')
flags.DEFINE_integer('GEN_ITERS', 50, '')
flags.DEFINE_integer('ITERATIONS_PER_SEQ_LENGTH', 15000, '')
flags.DEFINE_float('NOISE_STDEV', 10.0, '')
flags.DEFINE_integer('DISC_STATE_SIZE', 512, '')
flags.DEFINE_integer('GEN_STATE_SIZE', 512, '')
flags.DEFINE_boolean('TRAIN_FROM_CKPT', False, '')
flags.DEFINE_integer('GEN_GRU_LAYERS', 1, '')
flags.DEFINE_integer('DISC_GRU_LAYERS', 1, '')
flags.DEFINE_integer('START_SEQ', 1, '')
flags.DEFINE_integer('END_SEQ', 32, '')
flags.DEFINE_bool('PADDING_IS_SUFFIX', False, '')
flags.DEFINE_integer('SAVE_CHECKPOINTS_EVERY', 25000, '')
flags.DEFINE_boolean('LIMIT_BATCH', True, '')
flags.DEFINE_boolean('SCHEDULE_ITERATIONS', False, '')
flags.DEFINE_integer('SCHEDULE_MULT', 2000, '')
flags.DEFINE_boolean('DYNAMIC_BATCH', False, '')
flags.DEFINE_string('SCHEDULE_SPEC', 'all', '')

# Only for inference mode

flags.DEFINE_string('INPUT_SAMPLE', './output/sample.txt', '')


FLAGS = flags.FLAGS

# only for backward compatability

LOGS_DIR = os.path.join(FLAGS.LOGS_DIR,
                        "%s-%s-%s-%s-%s-%s-%s-" % (FLAGS.GENERATOR_MODEL, FLAGS.DISCRIMINATOR_MODEL,
                                                        FLAGS.GEN_ITERS, FLAGS.CRITIC_ITERS,
                                                        FLAGS.DISC_STATE_SIZE, FLAGS.GEN_STATE_SIZE,
                                                        time.time()))


class RestoreConfig():
    def __init__(self):
        if FLAGS.TRAIN_FROM_CKPT:
            self.restore_dir = self.set_restore_dir(load_from_curr_session=False)
        else:
            self.restore_dir = self.set_restore_dir(load_from_curr_session=True)

    def set_restore_dir(self, load_from_curr_session=True):
        if load_from_curr_session:
            restore_dir = os.path.join(LOGS_DIR, 'checkpoint')
        else:
            restore_dir = FLAGS.CKPT_PATH
        return restore_dir

    def get_restore_dir(self):
        return self.restore_dir


def create_logs_dir():
    os.makedirs(LOGS_DIR)

restore_config = RestoreConfig()
DATA_DIR = FLAGS.DATA_DIR
BATCH_SIZE = FLAGS.BATCH_SIZE
CRITIC_ITERS = FLAGS.CRITIC_ITERS
LAMBDA = FLAGS.LAMBDA
MAX_N_EXAMPLES = FLAGS.MAX_N_EXAMPLES
PICKLE_PATH = FLAGS.PICKLE_PATH
PICKLE_LOAD = True
CKPT_PATH = FLAGS.CKPT_PATH
GENERATOR_MODEL = FLAGS.GENERATOR_MODEL
DISCRIMINATOR_MODEL = FLAGS.DISCRIMINATOR_MODEL
GEN_ITERS = FLAGS.GEN_ITERS