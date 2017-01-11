# coding=utf-8
import glob
import sys

sys.path.append("../")

import tensorflow as tf

from model.LSTM import LSTM
from common.train import train_has_batch
from converter import train_data_converter

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
    batch_size = 5
    sequence_length = 20
    input_size = 128
    label_size = 128
    cell_size_list = [128, 128]
    keep_prob = 0.9
    optimizer_function = tf.train.GradientDescentOptimizer(0.1)
    clip_norm = 3


def main(argv):
    config = Config
    midi_dir = "../midi/multi_train_midi"
    midi_files = glob.glob(midi_dir + "/*.midi") + glob.glob(midi_dir + "/*.mid")
    # 一つの楽器のみ使用しているデータを対象とする
    train_data = [train_data_converter.create_midi_dataset(filename, config.sequence_length)[0] for filename in
                  midi_files]
    with tf.Graph().as_default() as graph:
        with tf.variable_scope("model"):
            model = LSTM(True, config, last_activation_function=tf.nn.sigmoid)

        with tf.Session() as session:
            saver = tf.train.Saver(tf.trainable_variables())
            train_has_batch(session, model, train_data, 0, config.batch_size, saver, FLAGS.logdir, FLAGS.load_dir)


if __name__ == '__main__':
    tf.app.run()
