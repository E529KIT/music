import glob

import tensorflow as tf
import numpy as np
import sys
import os

root_path = "../../../"
sys.path.append(root_path)

from converter.train_data_converter import create_midi_train_data_set_v2, generate_midi_v2
from model.LSTMSplit import Model as SplitModel

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("logdir", None, "log dir")
tf.app.flags.DEFINE_string("save_dir", None, "save dir")



class Config:
    batch_size = 1
    pitch_size = 128
    sequence_length = 1
    bar_size = 32
    pitch_cell_size_list = [128, 64]
    bar_cell_size_list = [128, 32]
    keep_prob = 0.7
    optimizer_function = tf.train.AdamOptimizer(0.01)
    clip_norm = 3
    cnn_out_size = 14


def main(argv):
    if FLAGS.logdir is None:
        print 'please set log dir'
        return
    if FLAGS.save_dir is None:
        print 'please set save dir'
        return

    config = Config
    pitch_size = config.pitch_size
    bar_size = config.bar_size
    midi_dir = root_path + "midi/multi_train_midi"
    midi_files = glob.glob(midi_dir + "/*.midi") + glob.glob(midi_dir + "/*.mid")
    train_data = create_midi_train_data_set_v2(midi_files, config.sequence_length, config.pitch_size, config.bar_size)
    model = SplitModel(config, generate=True)

    with tf.Session() as session:
        saver = tf.train.Saver(tf.trainable_variables())
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.logdir)
        saver.restore(session, checkpoint_file)
        for midi_file, data in zip(midi_files, train_data):
            inputs = data[0]
            init_threshold_value = 0.8
            down_threshold_value = 0.1

            pitch_state, bar_state = session.run([model.pitch_initial_state, model.bar_initial_state])
            result = [inputs[0][0]]
            for _ in range(100):
                feed_dict = {model.inputs: [[result[-1]]], model.pitch_initial_state: pitch_state}
                pitch_logits, pitch_state = session.run([model.pitch_logits, model.pitch_last_state], feed_dict)
                # calc pitch
                threshold_value = init_threshold_value
                while True:
                    pitch = map(lambda x: 1 if x >= threshold_value else 0, pitch_logits[0][0])
                    if max(pitch) == 1:
                        break
                    threshold_value -= down_threshold_value
                # calc bar
                feed_dict = {model.inputs: [[result[-1]]], model.bar_initial_state: bar_state,
                             model.pitch_labels: [[pitch]]}
                bar_logits, bar_state = session.run([model.bar_logits, model.bar_last_state], feed_dict)
                print bar_logits
                bar_max_index = np.argmax(bar_logits)
                bar = [0] * bar_size
                bar[bar_max_index] = 1
                result.append(pitch + bar)

            save_file_name = FLAGS.save_dir + "/" + os.path.basename(midi_file)
            print save_file_name
            generate_midi_v2(save_file_name, result, config.pitch_size, config.bar_size)


if __name__ == '__main__':
    tf.app.run()
