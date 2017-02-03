import glob
import sys
import os

import numpy as np
import tensorflow as tf

sys.path.append("../../")

from converter.train_data_converter import create_midi_train_data_set_v2
from model.LSTM_music import LSTM
from launcher.common.generate import  generate
from converter.train_data_converter import generate_midi_v2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("logdir", "log/cnn", "log dir")
tf.app.flags.DEFINE_string("save_dir", "generated", "save dir for generated midi")


class Config:
    batch_size = 1
    sequence_length = 1
    pitch_size = 128
    bar_size = 32
    cell_size_list = [160, 160]
    keep_prob = 1
    optimizer_function = tf.train.GradientDescentOptimizer(0.1)
    clip_norm = 3
    pitch_loss_wight = 1

def main(argv):
    config = Config
    midi_dir = "../../midi/multi_train_midi"
    midi_files = glob.glob(midi_dir + "/*.midi") + glob.glob(midi_dir + "/*.mid")
    train_data = create_midi_train_data_set_v2(midi_files, config.sequence_length, config.pitch_size, config.bar_size)
    model = LSTM(config, generate=True)
    pitch_size = config.pitch_size
    bar_size = config.bar_size

    with tf.Session() as session:
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(session, FLAGS.logdir + "/data/model")
        for midi_file, data in zip(midi_files, train_data):
            inputs = data[0]
            threshold_value = 0.5
            def norm_func(logits):
                pitch = map(lambda x: 1 if x > threshold_value else 0, logits[:pitch_size])
                bar_max_index = np.argmax(logits[pitch_size:])
                bar = [0] * bar_size
                bar[bar_max_index] = 1
                return pitch + bar

            generated_data = generate(session, model, inputs[:1], len(inputs), norm_func)
            save_file_name = FLAGS.save_dir + "/" + os.path.basename(midi_file)
            print save_file_name
            generate_midi_v2(save_file_name, generated_data, config.pitch_size, config.bar_size)



if __name__ == '__main__':
    tf.app.run()
