import glob
import sys
import tensorflow as tf

root_path = "../../../"
sys.path.append(root_path)

from model.LSTMContainedMusicTheory import Model
from model.LSTMSplit import Model as SplitModel
from launcher.common.sequence_example import get_padded_batch

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("logdir", None, "log dir")
tf.app.flags.DEFINE_string('model', None, "select model from normal or split")


class Config:
    batch_size = 40
    sequence_length = 100
    pitch_size = 128
    bar_size = 32
    cell_size_list = [160, 80]
    keep_prob = 0.7
    optimizer_function = tf.train.GradientDescentOptimizer(0.1)
    clip_norm = 3
    pitch_loss_wight = 20


class SplitConfig:
    batch_size = 40
    sequence_length = 100
    pitch_size = 128
    bar_size = 32
    pitch_cell_size_list = [128, 64]
    bar_cell_size_list = [32, 32]
    keep_prob = 0.7
    optimizer_function = tf.train.GradientDescentOptimizer(0.1)
    clip_norm = 3
    pitch_loss_wight = 20
    cnn_out_size = 14


def main(argv):
    tf.logging.set_verbosity('INFO')
    if FLAGS.logdir is None:
        tf.logging.ERROR("please select log dir")
        return
    with tf.Graph().as_default() as graph:
        if FLAGS.model == "normal":
            config = Config
            file_list = glob.glob("./train_data/*")
            input_size = output_size = config.pitch_size + config.bar_size
            inputs, labels, _ = get_padded_batch(file_list, config.batch_size, input_size, output_size,
                                                 config.sequence_length)
            model = Model(config, inputs=inputs, labels=labels, activate_function=tf.nn.sigmoid)
        elif FLAGS.model == "split":
            config = SplitConfig
            file_list = glob.glob("./train_data/*")
            input_size = output_size = config.pitch_size + config.bar_size
            inputs, labels, _ = get_padded_batch(file_list, config.batch_size, input_size, output_size,
                                                 config.sequence_length)
            model = SplitModel(config, inputs, labels)
        else:
            tf.logging.ERROR("please select model")
            return

    sv = tf.train.Supervisor(graph=graph, logdir=FLAGS.logdir, save_model_secs=30,
                             global_step=model.global_step, save_summaries_secs=60)
    with sv.managed_session() as session:
        global_step = session.run(model.global_step)
        while True:
            if sv.should_stop(): break
            if global_step % 100 == 0:
                (global_step, loss_, _) = session.run([model.global_step, model.loss,
                                                       model.train_optimizer])
                tf.logging.info('Global Step: %d - '
                                'Loss: %.3f - ',
                                global_step, loss_)
            else:
                _, global_step = session.run([model.train_optimizer, model.global_step])


if __name__ == '__main__':
    tf.app.run()
