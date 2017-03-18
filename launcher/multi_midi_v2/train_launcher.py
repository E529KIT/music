import glob
import sys

import tensorflow as tf

sys.path.append("../../")

from model.LSTMContainedMusicTheory import Model
from launcher.common.sequence_example import get_padded_batch

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("logdir", "log/1", "log dir")
tf.app.flags.DEFINE_string('load_dir', None, "load variables log dir")


class Config:
    batch_size = 20
    sequence_length = 40
    pitch_size = 128
    bar_size = 32
    cell_size_list = [160, 80]
    keep_prob = 0.7
    optimizer_function = tf.train.GradientDescentOptimizer(0.1)
    clip_norm = 3
    pitch_loss_wight = 20


def main(argv):
    with tf.Graph().as_default() as graph:
        config = Config
        file_list = glob.glob("./train_data/*")
        inputs, labels, _ = get_padded_batch(file_list, config.batch_size, 160, 160, config.sequence_length)
        model = Model(config, inputs=inputs, labels=labels)

    sv = tf.train.Supervisor(graph=graph, logdir=FLAGS.logdir, save_model_secs=30,
                             global_step=model.global_step)
    with sv.managed_session() as session:
        while (True):
            session.run(model.train_optimizer)


if __name__ == '__main__':
    tf.app.run()
