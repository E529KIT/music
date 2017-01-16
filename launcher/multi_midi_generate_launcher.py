# coding=utf-8
import glob
import os
import sys

import tensorflow as tf

sys.path.append("../")

from converter import train_data_converter
from model import LSTM


def generate(session, model, start_inputs, size, threshold_value):
    state = session.run(model.initial_state)
    fetches = [model.logits, model.last_state]
    result = [start_inputs[0][0]]
    logits = []
    start_input_size = len(start_inputs)

    for input_ in start_inputs:
        feed_dict = {model.inputs: [input_], model.initial_state: state}
        logits, state = session.run(fetches, feed_dict)
        logits = map(lambda x: 1 if x > threshold_value else 0, logits[0])
        result.append(logits)

    for i in range(size - start_input_size):
        feed_dict = {model.inputs: [[logits]], model.initial_state: state}
        logits, state = session.run(fetches, feed_dict)
        logits = map(lambda x: 1 if x > threshold_value else 0, logits[0])
        result.append(logits)
    return result


class Config:
    batch_size = 1
    sequence_length = 1
    input_size = 128
    label_size = 128
    cell_size_list = [128, 128]
    keep_prob = 1.0
    optimizer_function = tf.train.GradientDescentOptimizer(0.1)
    clip_norm = 3


if __name__ == '__main__':
    logdir = "../train_log/multi"
    save_dir = "./generated"
    midi_dir = "../midi/multi_train_midi"

    config = Config

    midi_files = glob.glob(midi_dir + "/*.midi") + glob.glob(midi_dir + "/*.mid")
    print midi_files
    # 一つの楽器のみ使用しているデータを対象とする
    train_data = [train_data_converter.create_midi_dataset(filename, config.sequence_length)[0] for filename in
                  midi_files]
    with tf.Graph().as_default() as graph:
        with tf.variable_scope("model"):
            model = LSTM.LSTM(True, config, True, tf.nn.sigmoid)

        with tf.Session() as session:
            saver = tf.train.Saver(tf.trainable_variables())
            session.run(tf.global_variables_initializer())
            saver.restore(session, logdir + "/data/model")
            for one_train_data, filename in zip(train_data, midi_files):
                train_data_size = len(one_train_data[0])
                buf = generate(session, model, one_train_data[0][:int(train_data_size / 2)], len(one_train_data[0]), 0.5)
                save_file_name = save_dir + "/" + os.path.basename(filename)
                print save_file_name
                train_data_converter.generate_midi(save_file_name, buf)