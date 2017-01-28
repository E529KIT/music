# coding=utf-8
import numpy as np
import complex_converter, wave_converter, midi_converter


def create_dataset(filename_list, sequence_length, sampling_rate):
    return [create_data(filename, sequence_length, sampling_rate) for filename in filename_list]


def create_data(filename, sequence_length, sampling_rate):
    data, params = wave_converter.load_wave(filename)
    data = wave_converter.fourier_transform(data, sampling_rate)
    if len(data[-1]) != sampling_rate:
        data.pop()  # 音楽の時間が小数点秒の場合、データを食わせられないため

    data = [a[:sampling_rate / 2] for a in data]

    data = [complex_converter.complex_list_to_double_list(one_data) for one_data in data]

    data = np.array(data)
    data /= 5000
    return _div_inputs_and_label(data, sequence_length)


def generate(filename, sampling_rate, data):
    data = np.array(data)
    data *= 5000
    data = data.tolist()
    data = [complex_converter.double_list_to_complex_list(one_data) for one_data in data]
    data = [one_data + ([complex(0, 0)] * (sampling_rate / 2)) for one_data in data]
    data = wave_converter.inverse_fourier_transform(data)
    print data[0]
    wave_converter.write_wave(filename, sampling_rate, data)


def _div_inputs_and_label(data, sequence_length, input_sequence_length=None):
    '''
    dataをinputsとlabelsに分けて、sequence_lengthの配列に変換する
    余った部分は０うめする。
    :param data: shape[N, input_size]
    :param sequence_length:
    :return: inputs, labels (shape:[?, sequence_length, input_size])
    '''
    if isinstance(data, list):
        data = np.array(data)
    if len(data) == 0:
        raise Exception("data.size == 0")
    if sequence_length == 1:
        surplus_size = 1
    else:
        surplus_size = sequence_length - len(data) % sequence_length + 1
    data = np.append(data, np.zeros([surplus_size, data.shape[1]]), axis=0)
    if sequence_length == 1:
        dataset_size = len(data) - 1
    else:
        dataset_size = len(data) / sequence_length
    inputs = data[:-1].reshape(dataset_size, sequence_length, data.shape[1])
    if input_sequence_length:
        inputs = [input_[:input_sequence_length] for input_ in inputs]
    labels = data[1:].reshape(dataset_size, sequence_length, data.shape[1])
    return inputs, labels


def create_midi_dataset(filename, sequence_length, input_sequence_length=None):
    midi = midi_converter.load_file(filename)
    dataset = midi_converter.convert_PrettyMIDI_to_train_data(midi)
    return [_div_inputs_and_label(one_data, sequence_length, input_sequence_length) for one_data in dataset]


def generate_midi(filename, data):
    '''
    0.5以上のものを1それ以下のものを0に変え、そのデータからmidiを作成する。
    :param filename:
    :param data:
    :return: None
    '''
    data = map(lambda one_data: map(lambda x: 1 if x > 0.5 else 0, one_data), data)
    midi_converter.save_file(filename, data)


def create_midi_train_data_set_v2(file_name_list, sequence_length, pitch_size, bar_size):
    return [_create_midi_train_data_v2(file_name, sequence_length, pitch_size, bar_size) for file_name in
            file_name_list]


def _create_midi_train_data_v2(file_name, sequence_length, pitch_size, bar_size):
    midi = midi_converter.load_file(file_name)
    # midiの中に複数楽器ある場合でも最初の一つのみ選択する。
    train_data = midi_converter.convert_PrettyMIDI_to_train_data(midi, False, bar_size)[0]

    train_data_v2 = []
    zero_time = 0
    for one_data in train_data:
        if sum(one_data) == 0:
            zero_time += 1
            continue

        if len(train_data_v2) > 1:
            # 次の音が鳴り始めるまで一生節以上あった場合も、なり始めを一生節後とする。
            zero_time = min([zero_time, bar_size - 1])
            train_data_v2[-1][pitch_size + zero_time] = 1

        train_data_v2.append(np.zeros([pitch_size + bar_size]))
        for pitch, on_off in enumerate(one_data):
            if on_off == 1: train_data_v2[-1][pitch] = 1
        zero_time = 0

    return _div_inputs_and_label(train_data_v2, sequence_length)


if __name__ == '__main__':
    load_filename = "/home/tatsuya/Music/1.wav"
    dataset = create_dataset([load_filename], 20, 8000)
    inputs, labels = dataset[0]
    inputs = np.absolute(inputs)
    print inputs.max()
    print inputs.min()
    # print max(max(max(inputs)))
    # print min(min(min(inputs)))
