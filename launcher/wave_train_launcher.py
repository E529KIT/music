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
    # 8KHz / 2 * 2 (sampling rate / 2 * complex)
    input_size = 8000
    label_size = 8000
    cell_size_list = [4000]
    keep_prob = 0.9
    optimizer_function = tf.train.GradientDescentOptimizer(0.1)
    clip_norm = 3


if __name__ == '__main__':
    with tf.Graph().as_default() as graph:
        config = Config

        load_filename = "/home/tatsuya/Music/1.wav"
        dataset = train_data_converter.create_dataset([load_filename], config.sequence_length, 8000)
        inputs, labels = dataset[0]

        with tf.variable_scope("model"):
            model = LSTM.LSTM(True, config, last_activation_function=tf.nn.tanh)

        with tf.Session() as session:
            saver = tf.train.Saver()
            train(session, model, inputs, labels, 1000, saver, FLAGS.logdir, FLAGS.load_dir)
