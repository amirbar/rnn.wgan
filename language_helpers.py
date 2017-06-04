import collections
import string

import numpy as np

from config import BATCH_SIZE


def tokenize_string(sample):
    return tuple(sample.lower().split(' '))


class NgramLanguageModel(object):
    def __init__(self, n, samples, tokenize=False):
        if tokenize:
            tokenized_samples = []
            for sample in samples:
                tokenized_samples.append(tokenize_string(sample))
            samples = tokenized_samples

        self._n = n
        self._samples = samples
        self._ngram_counts = collections.defaultdict(int)
        self._total_ngrams = 0
        for ngram in self.ngrams():
            self._ngram_counts[ngram] += 1
            self._total_ngrams += 1

    def ngrams(self):
        n = self._n
        for sample in self._samples:
            for i in range(len(sample) - n + 1):
                yield sample[i:i + n]

    def unique_ngrams(self):
        return set(self._ngram_counts.keys())

    def log_likelihood(self, ngram):
        if ngram not in self._ngram_counts:
            return -np.inf
        else:
            return np.log(self._ngram_counts[ngram]) - np.log(self._total_ngrams)

    def kl_to(self, p):
        # p is another NgramLanguageModel
        log_likelihood_ratios = []
        for ngram in p.ngrams():
            log_likelihood_ratios.append(p.log_likelihood(ngram) - self.log_likelihood(ngram))
        return np.mean(log_likelihood_ratios)

    def cosine_sim_with(self, p):
        # p is another NgramLanguageModel
        p_dot_q = 0.
        p_norm = 0.
        q_norm = 0.
        for ngram in p.unique_ngrams():
            p_i = np.exp(p.log_likelihood(ngram))
            q_i = np.exp(self.log_likelihood(ngram))
            p_dot_q += p_i * q_i
            p_norm += p_i ** 2
        for ngram in self.unique_ngrams():
            q_i = np.exp(self.log_likelihood(ngram))
            q_norm += q_i ** 2
        return p_dot_q / (np.sqrt(p_norm) * np.sqrt(q_norm))

    def precision_wrt(self, p):
        # p is another NgramLanguageModel
        num = 0.
        denom = 0
        p_ngrams = p.unique_ngrams()
        for ngram in self.unique_ngrams():
            if ngram in p_ngrams:
                num += self._ngram_counts[ngram]
            denom += self._ngram_counts[ngram]
        return float(num) / denom

    def recall_wrt(self, p):
        return p.precision_wrt(self)

    def js_with(self, p):
        log_p = np.array([p.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_p_m = np.sum(np.exp(log_p) * (log_p - log_m))

        log_p = np.array([p.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_q_m = np.sum(np.exp(log_q) * (log_q - log_m))

        return 0.5 * (kl_p_m + kl_q_m) / np.log(2)


def replace_trash(unicode_string):
    printable = set(string.printable)
    return filter(lambda x: x in printable, unicode_string)


def load_dataset(max_length, max_n_examples, tokenize=False, max_vocab_size=2048,
                 data_dir='/home/ishaan/data/1-billion-word-language-modeling-benchmark-r13output',
                 pad=True, dataset='training'):
    assert dataset == 'training' or dataset == 'heldout', "only available datasets are 'training' and 'heldout'"
    lines = []

    finished = False
    number_of_divided_files = 100 if dataset == 'training' else 50

    for i in range(number_of_divided_files-1):
        path = data_dir + ("/{}-monolingual.tokenized.shuffled/news.en{}-{}-of-{}".format(dataset,
                                                                                          '' if dataset == 'training' else '.heldout',
                                                                                          str(i + 1).zfill(5),
                                                                                          str(number_of_divided_files).zfill(5)))
        with open(path, 'r') as f:
            for line in f:
                line = line[:max_length]

                if tokenize:
                    line = tokenize_string(line)
                else:
                    line = tuple(line)

                if len(line) > max_length:
                    line = line[:max_length]

                if pad:
                    line = line + (("`",) * (max_length - len(line)))

                lines.append(line)

                if len(lines) == max_n_examples:
                    finished = True
                    break
        if finished:
            break

    np.random.shuffle(lines)

    import collections
    counts = collections.Counter(char for line in lines for char in line)

    charmap = {'unk': 0}
    inv_charmap = ['unk']

    for char, count in counts.most_common(10000000):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                filtered_line.append('unk')
        filtered_lines.append(tuple(filtered_line))

    # for i in range(100):
    #     print(filtered_lines[i])

    print("loaded {} lines in dataset".format(len(lines)))
    return filtered_lines, charmap, inv_charmap


def generate_argmax_samples_and_gt_samples(session, inv_charmap, fake_inputs, disc_fake, gen, real_inputs_discrete, feed_gt=True):
    scores = []
    samples = []
    samples_probabilites = []
    for i in range(10):
        argmax_samples, real_samples, samples_scores = generate_samples(session, inv_charmap, fake_inputs, disc_fake,
                                                                        gen, real_inputs_discrete, feed_gt=feed_gt)
        samples.extend(argmax_samples)
        scores.extend(samples_scores)
        samples_probabilites.extend(real_samples)
    return samples, samples_probabilites, scores


def generate_samples(session, inv_charmap, fake_inputs, disc_fake, gen, real_inputs_discrete, feed_gt=True):
    # sampled data here is only to calculate loss
    if feed_gt:
        f_dict = {real_inputs_discrete: next(gen)}
    else:
        f_dict = {}

    fake_samples, fake_scores = session.run([fake_inputs, disc_fake], feed_dict=f_dict)
    fake_scores = np.squeeze(fake_scores)

    decoded_samples = decode_indices_to_string(np.argmax(fake_samples, axis=2), inv_charmap)
    return decoded_samples, fake_samples, fake_scores


def decode_indices_to_string(samples, inv_charmap):
    decoded_samples = []
    for i in range(len(samples)):
        decoded = []
        for j in range(len(samples[i])):
            decoded.append(inv_charmap[samples[i][j]])
        decoded_samples.append(tuple(decoded))
    return decoded_samples


def inf_train_gen(lines, charmap):
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines) - BATCH_SIZE + 1, BATCH_SIZE):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i + BATCH_SIZE]],
                dtype='int32'
            )
