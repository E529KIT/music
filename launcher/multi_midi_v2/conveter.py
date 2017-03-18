import sys

sys.path.append("../../")
import glob
import os

import tensorflow as tf
from converter.train_data_converter import create_midi_train_data_set_with_file_name_v2
from launcher.common.sequence_example import create_sequence_example


def main(argv):
    sequence_length = 100
    pitch_size = 128
    bar_size = 32
    midi_dir = "../../midi/classic/BACH"
    midi_files = glob.glob(midi_dir + "/*.midi") + glob.glob(midi_dir + "/*.mid")
    train_data_sets, midi_file_names = create_midi_train_data_set_with_file_name_v2(midi_files, sequence_length,
                                                                                    pitch_size, bar_size)
    for train_data_set, midi_file_name in zip(train_data_sets, midi_file_names):
        writer = tf.python_io.TFRecordWriter("./train_data/" + os.path.basename(midi_file_name)[:-5])
        inputs_sequences, labels_sequences = train_data_set
        for inputs, labels in zip(inputs_sequences, labels_sequences):
            sequence_example = create_sequence_example(inputs, labels)
            writer.write(sequence_example.SerializeToString())
        writer.close()
        print midi_file_name


if __name__ == '__main__':
    tf.app.run()
