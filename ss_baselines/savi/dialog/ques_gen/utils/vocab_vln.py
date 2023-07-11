# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2019, Ranjay Krishna
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

"""Creates a vocabulary using iq_dataset for the vqa dataset.
"""

from collections import Counter
from ss_baselines.savi.dialog.ques_gen.utils import train_utils

import argparse
import json
import logging
import nltk
import numpy as np
import re
import sys


def process_text(text, vocab, max_length=20):
    """Converts text into a list of tokens surrounded by <start> and <end>.

    Args:
        text: String text.
        vocab: The vocabulary instance.
        max_length: The max allowed length.

    Returns:
        output: An numpy array with tokenized text.
        length: The length of the text.
    """
    tokens = tokenize(text.lower().strip())
    output = []
    output.append(vocab(vocab.SYM_SOQ))  # <start>
    output.extend([vocab(token) for token in tokens])
    output.append(vocab(vocab.SYM_EOS))  # <end>
    length = min(max_length, len(output))
    return np.array(output[:length]), length


def load_vocab(vocab_path):
    """Load Vocabulary object from a pickle file.

    Args:
        vocab_path: The location of the vocab pickle file.

    Returns:
    A Vocabulary object.
    """
    vocab = train_utils.Vocabulary()
    vocab.load(vocab_path)
    return vocab


def tokenize(sentence):
    """Tokenizes a sentence into words.

    Args:
        sentence: A string of words.

    Returns:
        A list of words.
    """
    # sentence = sentence.decode('utf8')
    if len(sentence) == 0:
        return []
    sentence = re.sub('\.+', r'.', sentence)
    sentence = re.sub('([a-z])([.,!?()])', r'\1 \2 ', sentence)
    sentence = re.sub('\s+', ' ', sentence)

    tokens = nltk.tokenize.word_tokenize(
            sentence.strip().lower())
    return tokens


def build_vocab(annot, threshold):
    """Build a vocabulary from the annotations.

    Args:
        annot: A list file containing the instruction with other annotation.
        threshold: The minimum number of times a word must occur. Otherwise it
            is treated as an `Vocabulary.SYM_UNK`.

    Returns:
        A Vocabulary object.
    """

    words = []
    counter = Counter()
    for i, entry in enumerate(annot):
        for inst in entry['instructions']:
            question = inst #.encode('utf8')
            q_tokens = tokenize(question)
            counter.update(q_tokens)

        if i % 1000 == 0:
            logging.info("worked on %d entries." % (i))

    # If a word frequency is less than 'threshold', then the word is discarded.
    words.extend([word for word, cnt in counter.items() if cnt >= threshold])
    words = list(set(words))
    vocab = create_vocab(words)
    return vocab


def create_vocab(words):
    # Adds the words to the vocabulary.
    vocab = train_utils.Vocabulary()
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters.
    parser.add_argument('--threshold', type=int, default=2,
                        help='Minimum word count threshold.')

    # Outputs.
    parser.add_argument('--vocab-path', type=str,
                        default='data/processed/vocab_iq_vln.json',
                        help='Path for saving vocabulary wrapper for vln.')
    args = parser.parse_args()

    # make sure to add symlink
    # for example:
    # ln -s /homes/supaul/Desktop/H/data/Fine-Grained-R2R /homes/supaul/sudipta/work/dialog_module/ques_gen/iq/data/raw
    tr_path = './data/raw/Fine-Grained-R2R/data/FGR2R_train.json'
    # tr is a list
    with open(tr_path) as f:
        tr = json.load(f)

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    vocab = build_vocab(tr, args.threshold)
    logging.info("Total vocabulary size: %d" % len(vocab))
    vocab.save(args.vocab_path)
    logging.info("Saved the vocabulary wrapper to '%s'" % args.vocab_path)
