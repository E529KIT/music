# coding=utf-8
import sys

import tensorflow as tf

sys.path.append("../")

from launcher.common.train import train
from converter import train_data_converter
from model import LSTM

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'logdir',
    "log/1",
    "log dir")
tf.app.flags.DEFINE_string(
    'load_dir',
    None,
    "load variables log dir")

class Config:
    batch_size = 1
    sequence_length = 20
    input_size = 128
    label_size = 128
    cell_size_list = [128, 128]
    keep_prob = 0.9
    optimizer_function = tf.train.GradientDescentOptimizer(0.1)
    clip_norm = 3


if __name__ == '__main__':
    logdir = FLAGS.logdir
    with tf.Graph().as_default() as graph:
        config = Config
        load_filename = "../midi/train_zanarukandonite.midi"
        dataset = train_data_converter.create_midi_dataset(load_filename, config.sequence_length)
        inputs, labels = dataset[0]

        with tf.variable_scope("model"):
            model = LSTM.LSTM(True, config, last_activation_function=tf.nn.sigmoid)

        with tf.Session() as session:
            saver = tf.train.Saver(tf.trainable_variables())
            train(session, model, inputs, labels, 1, saver, logdir, FLAGS.load_dir)
