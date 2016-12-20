# coding=utf-8
import sys
import tensorflow as tf
sys.path.append("../../")

from launcher.common.generate import generate
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


if __name__ == '__main__':
    with tf.Graph().as_default() as graph:
        config = Config
        load_filename = "/home/tatsuya/Music/1.wav"
        dataset = train_data_converter.create_dataset([load_filename], config.sequence_length, 8000)
        inputs, labels = dataset[0]

        with tf.variable_scope("model"):
            model = LSTM.LSTM(True, config, True, tf.nn.tanh)

        with tf.Session() as session:
            saver = tf.train.Saver()
            session.run(tf.global_variables_initializer())
            saver.restore(session, "data/1/model")
            buf = generate(session, model, inputs[0][0], 294)

        write_filename = "/home/tatsuya/Music/created.wave"
        train_data_converter.generate(write_filename, 8000, buf)
