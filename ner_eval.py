# -*- coding: utf-8 -*-

import argparse
import os

import shutil

from ner_build_data import CoNLLDataset
from ner_build_data import get_processing_word
from ner_build_data import get_trimmed_glove_vectors
from ner_build_data import load_vocab
from ner_define_model import NERModel

parser = argparse.ArgumentParser()

# general config
parser.add_argument('--output_path', default="results/ner/", type=str)

# dataset
parser.add_argument('--dev_filename', type=str, default="data/conll2003/eng.testa")
parser.add_argument('--test_filename', type=str, default='data/conll2003/eng.testb')
parser.add_argument('--train_filename', type=str, default='data/conll2003/eng.train')
parser.add_argument('--max_iter', default=0, type=int, help='if not None, max number of examples')
parser.add_argument('--rebuild_data', action='store_true', help='Rebuild dataset?')

# vocab (created from dataset with build_data.py)
parser.add_argument('--words_filename', type=str, default="data/conll2003/words.txt")
parser.add_argument('--tags_filename', type=str, default="data/conll2003/tags.txt")

# embeddings
parser.add_argument('--word_dim', default=300, type=int)
parser.add_argument('--glove_filename', type=str, default="/home/chaoming/fdisk2/data/NLP/glove/glove.6B.300d.txt")
parser.add_argument('--trimmed_filename', default='data/conll2003/glove.6B.300d.trimmed.npz', type=str,
                    help='trimmed embeddings (created from glove_filename with build_data.py)')

# training
parser.add_argument('--train_embeddings', action='store_true')
parser.add_argument('--nepochs', default=25, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--batch_size', default=20, type=int)
parser.add_argument('--lr_method', default="adam", type=str)
parser.add_argument('--lr', default=0.005, type=float)
parser.add_argument('--lr_decay', default=0.95, type=float)
parser.add_argument('--clip', default=-1, type=int, help='if negative, no clipping')
parser.add_argument('--nepoch_no_imprv', default=8, type=int)
parser.add_argument('--reload', action='store_true')
parser.add_argument('--random_seed', default=-1, type=int)

# model hyper parameters
parser.add_argument('--model_name', type=str, default='lstm')
parser.add_argument('--hidden_size', type=int, default=500)
parser.add_argument('--crf', action='store_true', help='Apply CRF?')
parser.set_defaults(crf=True, )

# others
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--merge', default='mean', type=str)
parser.add_argument('--att_size', default=0, type=int)

# parse
config = parser.parse_args()

# make dirs and create instance of logger
if os.path.exists(config.output_path):
    for filename in os.listdir(config.output_path):
        path = os.path.join(config.output_path, filename)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
else:
    os.makedirs(config.output_path)

if config.merge == 'attention' and config.att_size == 0:
    config.att_size = config.hidden

# output parameters
print("-" * 50)
print("Parameters:")
print("-" * 50)
for attr in dir(config):
    if not attr.startswith('_'):
        print("\t{:>20} = {:<10}".format(attr, str(getattr(config, attr))))
print()


def main():
    # load vocabs
    vocab_words = load_vocab(config.words_filename)
    vocab_tags = load_vocab(config.tags_filename)

    # get processing functions
    processing_word = get_processing_word(vocab_words, lowercase=True)
    processing_tag = get_processing_word(vocab_tags, lowercase=False)

    # get pre trained embeddings
    embeddings = get_trimmed_glove_vectors(config.trimmed_filename)

    # create dataset
    dev = CoNLLDataset(config.dev_filename, processing_word, processing_tag, config.max_iter)
    test = CoNLLDataset(config.test_filename, processing_word, processing_tag, config.max_iter)
    train = CoNLLDataset(config.train_filename, processing_word, processing_tag, config.max_iter)

    # build model
    model = NERModel(config, embeddings, ntags=len(vocab_tags))
    model.build()

    # train, evaluate and interact
    model.train(train, dev, vocab_tags, config.verbose)
    model.evaluate(test, vocab_tags)

    print("- dev  acc {:04.2f} - f1 {:04.2f}".format(100 * model.dev_res[0], 100 * model.dev_res[1]))
    print()


if __name__ == "__main__":
    # build_data()
    main()
