import sys

sys.path.append("../")

import numpy as np
import tensorflow as tf

from converter import midi_converter


def main(argv):
    origin_midi_file_name = "../midi/train_zanarukandonite.midi"
    generated_midi_file_name = "./test.mid"
    origin_midi = midi_converter.load_file(origin_midi_file_name)
    origin_data = midi_converter.convert_PrettyMIDI_to_train_data(origin_midi)

    generated_midi = midi_converter.load_file(generated_midi_file_name)
    generated_data = midi_converter.convert_PrettyMIDI_to_train_data(generated_midi)

    correct_prediction = [np.array_equal(origin_one_data, generated_one_data) for origin_one_data, generated_one_data in
                          zip(origin_data[0], generated_data[0])]
    data_size = len(correct_prediction)
    accuracy = np.sum(correct_prediction) / float(data_size)
    print data_size
    print accuracy
    failed_index_list = [i for i, correct in enumerate(correct_prediction) if not correct]
    print failed_index_list


if __name__ == '__main__':
    tf.app.run()
