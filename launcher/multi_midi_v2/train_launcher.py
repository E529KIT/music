import glob
import sys

import tensorflow as tf

sys.path.append("../../")

from converter.train_data_converter import create_midi_train_data_set_v2
from model.LSTM_music import LSTM
from launcher.common.train import train_has_batch

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("logdir", "log/1", "log dir")
tf.app.flags.DEFINE_string('load_dir', None, "load variables log dir")


class Config:
    batch_size = 10
    sequence_length = 32
    pitch_size = 128
    bar_size = 32
    cell_size_list = [160, 160]
    keep_prob = 0.5
    optimizer_function = tf.train.GradientDescentOptimizer(0.1)
    clip_norm = 3
    pitch_loss_wight = 20


def main(argv):
    config = Config
    midi_dir = "../../midi/classic/BACH"
    midi_files = glob.glob(midi_dir + "/*.midi") + glob.glob(midi_dir + "/*.mid")
    train_data = create_midi_train_data_set_v2(midi_files, config.sequence_length, config.pitch_size, config.bar_size)
    model = LSTM(config)

    with tf.Session() as session:
        saver = tf.train.Saver(tf.trainable_variables())
        train_has_batch(session, model, train_data, 0, config.batch_size, saver, FLAGS.logdir, FLAGS.load_dir)


if __name__ == '__main__':
    tf.app.run()
