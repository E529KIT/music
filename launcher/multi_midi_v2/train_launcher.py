import glob
import sys

import tensorflow as tf

sys.path.append("../../")

from launcher.common.train import train_has_batch
from model.LSTM_music import LSTM

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("logdir", "log/1", "log dir")
tf.app.flags.DEFINE_string('load_dir', None, "load variables log dir")


class Config:
    batch_size = 5
    sequence_length = 36
    pitch_size = 128
    bar_size = 32
    cell_size_list = [160, 160]
    keep_prob = 0.9
    optimizer_function = tf.train.AdamOptimizer(0.1)
    clip_norm = 3
    pitch_loss_wight = 1


def main(argv):
    config = Config
    midi_dir = "../midi/multi_train_midi"
    midi_files = glob.glob(midi_dir + "/*.midi") + glob.glob(midi_dir + "/*.mid")
    # TODO: create dataset from midis
    train_data = None
    with tf.variable_scope("model"):
        model = LSTM(config)

    with tf.Session() as session:
        saver = tf.train.Saver(tf.trainable_variables())
        train_has_batch(session, model, train_data, 0, config.bar_size, saver, FLAGS.logdir, FLAGS.load_dir)


if __name__ == '__main__':
    tf.app.run()
