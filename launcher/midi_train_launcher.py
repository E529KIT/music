# coding=utf-8
import sys

import tensorflow as tf

sys.path.append("../")

from launcher.common.train import train
from converter import train_data_converter
from model import LSTM


class Config:
    batch_size = 1
    sequence_length = 40
    input_size = 128
    label_size = 128
    cell_size_list = [128]
    keep_prob = 1.0
    optimizer_function = tf.train.AdamOptimizer(0.1)
    clip_norm = 3


if __name__ == '__main__':
    logdir = "log/1"
    with tf.Graph().as_default() as graph:
        config = Config

        load_filename = "../midi/bwv772.midi"
        dataset = train_data_converter.create_midi_dataset(load_filename, config.sequence_length)
        inputs, labels = dataset[0]

        with tf.variable_scope("model"):
            model = LSTM.LSTM(True, config, last_activation_function=tf.nn.tanh)

        with tf.Session() as session:
            saver = tf.train.Saver()
            train(session, model, inputs, labels, 1000, saver, logdir)
