import os
import tensorflow as tf

import model_and_data_serialization
from config import LOGS_DIR

FLAGS = tf.app.flags.FLAGS


def define_summaries(disc_cost, gen_cost, seq_length):
    train_writer = tf.summary.FileWriter(LOGS_DIR)
    tf.summary.scalar("d_loss_%d" % seq_length, disc_cost)
    tf.summary.scalar("g_loss_%d" % seq_length, gen_cost)
    merged = tf.summary.merge_all()
    return merged, train_writer


def log_samples(samples, scores, iteration, seq_length, prefix):
    sample_scores = zip(samples, scores)
    sample_scores = sorted(sample_scores, key=lambda sample: sample[1])


    with open(model_and_data_serialization.get_internal_checkpoint_dir(seq_length) + '/{}_samples_{}.txt'.format(
            prefix, iteration),
              'a') as f:
        for s, score in sample_scores:
            s = "".join(s)
            f.write("%s||\t%f\n" % (s, score))
    f.close()


def log_run_settings():
    with open(os.path.join(LOGS_DIR, 'run_settings.txt'), 'w') as f:
        for key in tf.flags.FLAGS.__flags.keys():
            entry = "%s: %s" % (key, tf.flags.FLAGS.__flags[key])
            f.write(entry + '\n')
            print(entry)
    f.close()


def get_grams_cached(lines):
    grams_filename = FLAGS.PICKLE_PATH + '/true-char-ngrams.pkl'
    if os.path.exists(grams_filename):
        return model_and_data_serialization.load_picklized(grams_filename)
    else:
        grams = get_grams(lines)
        model_and_data_serialization.save_picklized(grams, grams_filename)
        return grams


def get_grams(lines):
    lines_joined = [''.join(l) for l in lines]

    unigrams = dict()
    bigrams = dict()
    trigrams = dict()
    quadgrams = dict()
    token_count = 0

    for l in lines_joined:
        l = l.split(" ")
        l = filter(lambda x: x != ' ' and x != '', l)

        for i in range(len(l)):
            token_count += 1
            unigrams[l[i]] = unigrams.get(l[i], 0) + 1
            if i >= 1:
                bigrams[(l[i - 1], l[i])] = bigrams.get((l[i - 1], l[i]), 0) + 1
            if i >= 2:
                trigrams[(l[i - 2], l[i - 1], l[i])] = trigrams.get((l[i - 2], l[i - 1], l[i]), 0) + 1
            if i >= 3:
                quadgrams[(l[i - 3], l[i - 2], l[i - 1], l[i])] = quadgrams.get((l[i - 3], l[i - 2], l[i - 1], l[i]), 0) + 1

    return unigrams, bigrams, trigrams, quadgrams


def percentage_real(samples_grams, real_grams):
    grams_in_real = 0

    for g in samples_grams:
        if g in real_grams:
            grams_in_real += 1
    if len(samples_grams) > 0:
        return grams_in_real * 1.0 / len(samples_grams)
    return 0


def percentage_startswith(e_samples, unigrams_real):
    counter = 0
    for prefix in e_samples:
        for uni in unigrams_real:
            if uni.startswith(prefix):
                counter += 1
                break
    # print counter
    return counter * 1.0 / len(e_samples.keys())
