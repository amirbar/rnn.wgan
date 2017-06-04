import os
import cPickle as pickle

import tensorflow as tf

import language_helpers
from config import PICKLE_PATH, PICKLE_LOAD, DATA_DIR, restore_config

CHARMAP_FN = 'charmap_32.pkl'
INV_CHARMAP_FN = 'inv_charmap_32.pkl'
INV_CHARMAP_PKL_PATH = PICKLE_PATH + '/' + INV_CHARMAP_FN
CHARMAP_PKL_PATH = PICKLE_PATH + '/' + CHARMAP_FN

def load_picklized(path):
    with open(path, 'rb') as f:
        content = pickle.load(f)
        f.close()
    return content


def save_picklized(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_dataset_from_pkl(b_lines, b_charmap, b_inv_charmap, lines_pkl_path):
    if b_lines:
        lines = load_picklized(lines_pkl_path)
    else:
        lines = None

    if b_charmap:
        charmap = load_picklized(CHARMAP_PKL_PATH)
    else:
        charmap = None

    if b_inv_charmap:
        inv_charmap = load_picklized(INV_CHARMAP_PKL_PATH)
    else:
        inv_charmap = None

    return lines, charmap, inv_charmap


def load_dataset(b_lines=True, b_charmap=True, b_inv_charmap=True, seq_length=32, n_examples=10000000, tokenize=False,
                 pad=True, dataset='training'):
    LINES_FN = 'lines_%s_%s.pkl' % (seq_length, tokenize)
    if dataset != 'training':
        LINES_FN = dataset + '_' + LINES_FN
    LINES_PKL_PATH = PICKLE_PATH + '/' + LINES_FN

    if PICKLE_PATH is not None and PICKLE_LOAD is True and (
                    b_lines is False or (b_lines and os.path.exists(LINES_PKL_PATH))) \
            and (b_charmap is False or (b_charmap and os.path.exists(CHARMAP_PKL_PATH))) and \
            (b_inv_charmap is False or (b_inv_charmap and os.path.exists(INV_CHARMAP_PKL_PATH))):

        print("Loading lines, charmap, inv_charmap from pickle files")
        lines, charmap, inv_charmap = load_dataset_from_pkl(b_lines=b_lines, b_charmap=b_charmap,
                                                            b_inv_charmap=b_inv_charmap, lines_pkl_path=LINES_PKL_PATH)

    else:
        print("Loading lines, charmap, inv_charmap from Dataset & Saving to pickle")
        lines, charmap, inv_charmap = language_helpers.load_dataset(
            max_length=seq_length,
            max_n_examples=n_examples,
            data_dir=DATA_DIR,
            tokenize=tokenize,
            pad=pad,
            dataset=dataset
        )

        # save to pkl
        if not os.path.isdir(PICKLE_PATH):
            os.mkdir(PICKLE_PATH)

        if b_lines:
            save_picklized(lines, LINES_PKL_PATH)
        if b_charmap:
            save_picklized(charmap, CHARMAP_PKL_PATH)
        if b_inv_charmap:
            save_picklized(inv_charmap, INV_CHARMAP_PKL_PATH)

    return lines, charmap, inv_charmap


def get_internal_checkpoint_dir(seq_length):
    internal_checkpoint_dir = os.path.join(restore_config.get_restore_dir(), "seq-%d" % seq_length)
    if not os.path.isdir(internal_checkpoint_dir):
        os.makedirs(internal_checkpoint_dir)
    return internal_checkpoint_dir


def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
            else:
                print("Not loading: %s." % saved_var_name)
    saver = tf.train.Saver(restore_vars)
    print ("Restoring vars:")
    print (restore_vars)
    saver.restore(session, save_file)
