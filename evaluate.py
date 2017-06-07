import os

import tensorflow as tf

import language_helpers
import model_and_data_serialization
from summaries import get_grams, percentage_real

'''
example usage:
python evaluate.py --INPUT_SAMPLE=path/to/my/sample.txt --MAX_N_EXAMPLES=10000000
'''

FLAGS = tf.app.flags.FLAGS


def evaluate(input_sample, gt_grams):
    # char level evaluation

    sample_lines = load_sample(input_sample, tokenize=False)

    u_samples, b_samples, t_samples, q_samples = get_grams(sample_lines)
    u_real, b_real, t_real, q_real = gt_grams

    print "Unigrams: %f" % percentage_real(u_samples, u_real)
    print "Bigrams: %f" % percentage_real(b_samples, b_real)
    print "Trigrams: %f" % percentage_real(t_samples, t_real)
    print "Quad grams: %f" % percentage_real(q_samples, q_real)


def cut_end_words(lines):
    lines_without_ends = []
    for l in lines:
        lines_without_ends.append(l[:-1])

    return lines_without_ends


def load_sample(input_sample, tokenize=False):
    with open(input_sample, 'r') as f:
        lines_samples = [l for l in f]
    f.close()
    preprocessed_lines = preprocess_sample(lines_samples, '\n', tokenize=tokenize)
    return preprocessed_lines


def load_gt(tokenize=False, dataset='heldout'):
    # test on char level
    print("Loading lines, charmap, inv_charmap")
    lines, _, _ = model_and_data_serialization.load_dataset(
        b_lines=True,
        b_charmap=False,
        b_inv_charmap=False,
        n_examples=FLAGS.MAX_N_EXAMPLES,
        tokenize=tokenize,
        pad=False,
        dataset=dataset
    )

    return lines


def preprocess_sample(lines_samples, separator, tokenize):
    preprocessed_lines = []
    for line in lines_samples:
        if separator is not None:
            line = separator.join(line.split(separator)[:-1])

        if tokenize:
            line = language_helpers.tokenize_string(line)
        else:
            line = tuple(line)

        preprocessed_lines.append(line)
    return preprocessed_lines


def get_gt_grams_cached(lines, dataset='training'):
    grams_filename = 'true-char-ngrams.pkl'
    if dataset == 'heldout':
        grams_filename = 'heldout_' + grams_filename
    grams_filename = FLAGS.PICKLE_PATH + '/' + grams_filename
    if os.path.exists(grams_filename):
        return model_and_data_serialization.load_picklized(grams_filename)
    else:
        grams = get_grams(lines)
        model_and_data_serialization.save_picklized(grams, grams_filename)
        return grams


dataset = 'heldout'
lines = load_gt(tokenize=False, dataset=dataset)
gt_grams = get_gt_grams_cached(lines, dataset)
evaluate(FLAGS.INPUT_SAMPLE, gt_grams)