# coding=utf-8
import sys

import tensorflow as tf

sys.path.append("../")

from launcher.common.train import train
from converter import train_data_converter
from model import LSTM


class Config:
    batch_size = 1
    sequence_length = 20
    # 8KHz / 2 * 2 (sampling rate / 2 * complex)
    input_size = 8000
    label_size = 8000
    cell_size_list = [4000]
    keep_prob = 1.0
    optimizer_function = tf.train.AdamOptimizer(0.1)
    clip_norm = 3


if __name__ == '__main__':
    logdir = "data/1/"
    with tf.Graph().as_default() as graph:
        config = Config

        load_filename = "/home/tatsuya/Music/1.wav"
        dataset = train_data_converter.create_dataset([load_filename], config.sequence_length, 8000)
        inputs, labels = dataset[0]

        with tf.variable_scope("model"):
            model = LSTM.LSTM(True, config, last_activation_function=tf.nn.tanh)

        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, "data/1/model")
            train(session, model, inputs, labels, 1000, saver, "data/1")
