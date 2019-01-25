# -*- coding: utf-8 -*-

import argparse
import os
import sys
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from rnns import get_test_model_by_name
from tools import Progbar

parser = argparse.ArgumentParser()

# training parameters
parser.add_argument('--seed', default=0, type=int, help='Random seed, if 0, then it will be randomly initialized.')
parser.add_argument('--model', default='irnn', type=str, help='Model name.')
parser.add_argument('--hidden', default=200, type=int, help='Recurrent network hidden size.')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size.')
parser.add_argument('--nepoch_no_imprv', default=5, type=int, help='Epoch number applied to early stopping method.')
parser.add_argument('--dropout', default=0., type=float, help='Dropout rate.')
parser.add_argument('--max_norm', default=0., type=float, help='Max normalization.')
parser.add_argument('--optimizer', default='sgd', type=str, help='Optimizer')
parser.add_argument('--layer_num', default=1, type=int, help='RNN model layer numbers.')
parser.add_argument('--pool', default='mean', type=str, help='Pooling methods, including "mean", "max" and "last"')

parser.add_argument('--merge', default='mean', type=str)
parser.add_argument('--att_size', default=0, type=int)

parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate.')
parser.add_argument('--lr_decay', default=0.95, type=float, help='Learning rate decay every epoch.')
parser.add_argument('--epoches', default=80, type=int, help='Epoch number for training.')
parser.add_argument('--small_epoches', default=50, type=int, help='Epoch number for training.')

parser.add_argument('--shuffle_training_data', action='store_false')

# fixed Network Parameters
parser.add_argument('--num_input', default=28, type=int, help='MNIST data input (img shape: 28*28)')
parser.add_argument('--timesteps', default=28, type=int, help='timesteps')
parser.add_argument('--num_classes', default=10, type=int, help='MNIST total classes (0-9 digits)')

# other parameters
parser.add_argument('--verbose', action='store_false')
parser.add_argument('--save_path', default='results/mnist/')

args = parser.parse_args()

# check seed for initializing parameters
if args.seed == 0:
    args.seed = np.random.randint(10000)

if args.merge == 'attention' and args.att_size == 0:
    args.att_size = args.hidden


# output parameters
print("Parameters:")
for p in dir(args):
    if p.startswith('_'):
        continue
    print("\t{:>20} = {:<10}".format(p, getattr(args, p)))
print()
sys.stdout.flush()

# make dirs and create instance of logger
if os.path.exists(args.save_path):
    for filename in os.listdir(args.save_path):
        path = os.path.join(args.save_path, filename)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
else:
    os.makedirs(args.save_path)

rng_for_rng = np.random.RandomState(seed=args.seed)


class MnistModel(object):
    def __init__(self, is_training_mode=True):
        self.input = tf.placeholder(tf.float32, [None, args.timesteps * args.num_input], 'input')
        self.target = tf.placeholder(tf.float32, [None, args.num_classes], 'target')
        input = tf.reshape(self.input, (-1, args.timesteps, args.num_input))
        input = tf.unstack(input, args.timesteps, axis=1)

        # define rnn
        rnn_cell = get_test_model_by_name(args.model.lower(), args.hidden, merge=args.merge, att_size=args.att_size)
        if is_training_mode and 0. < args.dropout < 1.:
            rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=args.dropout)
        with tf.variable_scope('RNN-model'):
            outputs, states = tf.nn.static_rnn(rnn_cell, input, dtype=tf.float32)

        # define pooling methods
        if args.pool == 'mean':
            output = tf.reduce_mean(outputs, axis=0)
        elif args.pool == 'max':
            output = tf.reduce_max(outputs, axis=0)
        elif args.pool == 'last':
            output = outputs[-1]
        else:
            raise ValueError('Unknown pooling method: {}'.format(args.pool))

        # define softmax
        with tf.variable_scope('softmax'):
            softmax_w = tf.get_variable('softmax_w', [args.hidden, args.num_classes], dtype=tf.float32)
            softmax_b = tf.get_variable('softmax_b', [args.num_classes, ], dtype=tf.float32)
            logits = tf.matmul(output, softmax_w) + softmax_b

        # define loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.target))

        # define optimizer
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='lr')
        with tf.variable_scope('train_step'):
            if args.optimizer == 'adam':
                opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            elif args.optimizer == 'rmsprop':
                opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
            elif args.optimizer == 'sgd':
                opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            else:
                raise ValueError('Unknown optimizer: {}'.format(args.optimizer))

            if args.max_norm > 0.:
                gradients, variables = zip(*opt.compute_gradients(self.loss))
                gradients, global_norm = tf.clip_by_global_norm(gradients, args.max_norm)
                self.train_op = opt.apply_gradients(zip(gradients, variables))
            else:
                self.train_op = opt.minimize(self.loss)

        # define evaluation
        self.correct_pred = tf.cast(tf.equal(tf.argmax(logits, axis=1),
                                             tf.argmax(self.target, axis=1)),
                                    dtype=tf.int32)
        self.correct_num = tf.reduce_sum(self.correct_pred)
        self.accuracy = tf.reduce_mean(self.correct_pred)


def run_epoch(model, sess, data, is_training_mode=True):
    correct_nums = 0

    if is_training_mode and args.shuffle_training_data:
        rng_seed = rng_for_rng.randint(100, 100000)
        np.random.RandomState(rng_seed).permutation(data.images)
        np.random.RandomState(rng_seed).permutation(data.labels)

    bth = args.batch_size if is_training_mode else 200
    epoches = int(np.ceil(data.num_examples / bth))
    prog = Progbar(target=epoches, verbose=args.verbose and is_training_mode)

    for i in range(epoches):
        feed_dict = {
            model.input: data.images[bth * i: bth * i + bth],
            model.target: data.labels[bth * i: bth * i + bth],
            model.learning_rate: args.learning_rate,
        }
        if is_training_mode:
            _, loss, cnum = sess.run(fetches=[model.train_op, model.loss, model.correct_num],
                                     feed_dict=feed_dict)
        else:
            loss, cnum = sess.run(fetches=[model.loss, model.correct_num],
                                  feed_dict=feed_dict)
        prog.update(i + 1, [('train loss', loss)])
        correct_nums += cnum

    return correct_nums / float(data.num_examples)


def main():
    # load dataset
    mnist = input_data.read_data_sets('./data/mnist/', one_hot=True)

    # define Graph
    with tf.Graph().as_default():
        with tf.name_scope('Train'):
            with tf.variable_scope('Model', reuse=None):
                train_model = MnistModel(is_training_mode=True)

        with tf.name_scope("Test"):
            with tf.variable_scope('Model', reuse=True):
                test_model = MnistModel(is_training_mode=False)

        best_score = 0.
        nepoch_no_imprv = 0

        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as session:
            # Important! random seed
            tf.set_random_seed(args.seed)

            # Initialize variables
            session.run(tf.global_variables_initializer())

            for epoch in range(1, args.epoches+1):
                print('Epoch {:>2} out of {:>2}'.format(epoch, args.epoches))

                acc_t = run_epoch(train_model, session, mnist.train, True)
                acc_v = run_epoch(test_model, session, mnist.validation, False)

                # fine-tune learning rate
                if epoch >= args.small_epoches or acc_v < best_score:
                    args.learning_rate *= args.lr_decay

                runout = '- train acc: {:04.2f}; dev acc: {:04.2f}'.format(acc_t * 100, acc_v * 100)

                if acc_v > best_score:
                    best_score = acc_v
                    nepoch_no_imprv = 0
                    runout += '; new best score!'
                    saver.save(session, os.path.join(args.save_path, 'model.weights'))
                else:
                    nepoch_no_imprv += 1
                print(runout + '\n')
                sys.stdout.flush()

                if nepoch_no_imprv >= args.nepoch_no_imprv:
                    print("- early stopping {} epochs without improvement. \n".format(nepoch_no_imprv))
                    break

            saver.restore(session, os.path.join(args.save_path, 'model.weights'))
            acc_t = run_epoch(test_model, session, mnist.test, False)
            print("- test acc {:04.2f} \n".format(100 * acc_t))
            sys.stdout.flush()


if __name__ == '__main__':
    main()


