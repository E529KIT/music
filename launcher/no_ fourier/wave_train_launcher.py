# coding=utf-8
import datetime
import sys

import numpy as np
import tensorflow as tf

sys.path.append("../../")

from converter import wave_converter
from model import LSTM


class Config:
    batch_size = 1
    sequence_length = 200
    # 8KHz / 2 * 2 (sampling rate / 2 * complex)
    input_size = 256
    label_size = 256
    cell_size_list = [256, 256, 256]
    keep_prob = 1.0
    optimizer_function = tf.train.AdamOptimizer(0.1)


def create_dataset(raw_data, step, sequence_length):
    data = raw_data[step * sequence_length: (step + 1) * sequence_length + 1]
    if len(data) == 0:
        return None, None
    # to one hot
    dataset = np.zeros((len(data), 256))
    dataset[np.arange(len(data)), data] = 1

    zero_data = [0] * 256
    zero_data[125] = 1

    surplus_size = sequence_length - len(dataset) + 1
    if surplus_size == 0:
        inputs = dataset[:-1]
        labels = dataset[1:]
    else:
        inputs = np.append(dataset[:-1], [[zero_data] * surplus_size])
        labels = np.append(dataset[1:], [[zero_data] * (surplus_size + 1)])
    return inputs, labels


def train(session, model, raw_data, sequence_length, epoch, saver, savepath):
    train_writer = tf.train.SummaryWriter('./log', session.graph)
    merged = tf.merge_all_summaries()
    session.run(tf.global_variables_initializer())
    global_step, state = session.run([model.global_step, model.initial_state])
    fetches = [model.train_optimizer, model.global_step, model.last_state, model.loss, merged]
    for i in range(epoch):
        counter = 0
        while True:
            input_, label = create_dataset(raw_data, counter, sequence_length)
            if input_ is None: break
            feed_dict = {model.inputs: [input_], model.initial_state: state, model.labels: [label]}
            _, global_step, state, loss, summaries = session.run(fetches, feed_dict)
            train_writer.add_summary(summaries, global_step)
        if i % 50 == 0:
            saver.save(session, savepath)
            now = datetime.datetime.today()
            print '%d: %s' % (i, now.isoformat())


if __name__ == '__main__':
    with tf.Graph().as_default() as graph:
        config = Config

        load_filename = "/home/tatsuya/Music/1.wav"
        raw_data, sampling_rate = wave_converter.load_wave(load_filename)

        with tf.variable_scope("model"):
            model = LSTM.LSTM(True, config, last_activation_function=tf.nn.softmax)

        savepath = "data/1/model"
        with tf.Session() as session:
            saver = tf.train.Saver()
            # saver.restore(session, savepath)
            train(session, model, raw_data, config.sequence_length, 1000, saver, savepath)
