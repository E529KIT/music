import sys

import tensorflow as tf

sys.path.append("../")

from launcher.common.generate import generate
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
            model = LSTM.LSTM(True, config, True, tf.nn.tanh)

        with tf.Session() as session:
            saver = tf.train.Saver()
            session.run(tf.global_variables_initializer())
            saver.restore(session, logdir + "/data/model")
            buf = generate(session, model, inputs[0][0], len(inputs))

        save_filename = "test.mid"
        train_data_converter.generate_midi(save_filename, buf)
