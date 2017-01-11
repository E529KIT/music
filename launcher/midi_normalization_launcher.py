import glob
import os
import sys

sys.path.append("../")
import tensorflow as tf
from converter import midi_converter


def main(argv):
    save_dir = "../midi/multi_train_midi"
    midi_dir = "../midi/multi_train_raw_midi"
    midi_file_names = glob.glob(midi_dir + "/*.midi") + glob.glob(midi_dir + "/*.mid")
    for midi_file_name in midi_file_names:
        midi = midi_converter.load_file(midi_file_name)
        data = midi_converter.convert_PrettyMIDI_to_train_data(midi)[0]
        start_time = 0
        for j, one_data in enumerate(data):
            if not sum(one_data) == 0:
                start_time = j
                break
        data = data[start_time:]
        save_file_name = save_dir + "/" + os.path.basename(midi_file_name)
        midi_converter.save_file(save_file_name, data)


if __name__ == '__main__':
    tf.app.run()
