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

    # 訓練データの縮小のため可聴範囲が外の音は消す
    one_data_size = sampling_rate / 2 * 2
    data = [a[:sampling_rate / 2] for a in data]

    data = [complex_converter.complex_list_to_double_list(one_data) for one_data in data]

    data = np.array(data)
    data /= 5000
    data = data.tolist()
    data_set_size = len(data)
    inputs = [data[i:i + sequence_length]
              for i in range(0, data_set_size, sequence_length)]
    labels = [data[i: i + sequence_length]
              for i in range(1, data_set_size, sequence_length)]

    # 余剰データを0でうめる
    surplus_size = sequence_length - (data_set_size % sequence_length)
    # lne(inputs[-1]) == len(labels[-1]) + 1
    zero_list = [[0] * one_data_size] * (surplus_size + 1)
    inputs[-1] = inputs[-1] + zero_list[: -1]
    labels[-1] = labels[-1] + zero_list
    return inputs, labels


def generate(filename, sampling_rate, data):
    data = np.array(data)
    data *= 5000
    data = data.tolist()
    data = [complex_converter.double_list_to_complex_list(one_data) for one_data in data]
    data = [one_data + ([complex(0, 0)] * (sampling_rate / 2)) for one_data in data]
    data = wave_converter.inverse_fourier_transform(data)
    print data[0]
    wave_converter.write_wave(filename, sampling_rate, data)


def _div_inputs_and_label(data, sequence_length):
    '''
    dataをinputsとlabelsに分けて、sequence_lengthの配列に変換する
    余った部分は０うめする。
    :param data: shape[N, input_size]
    :param sequence_length:
    :return: inputs, labels (shape:[?, sequence_length, input_size])
    '''
    data = np.array(data)
    if len(data) == 0:
        raise Exception("data.size == 0")
    surplus_size = sequence_length - len(data) % sequence_length + 1
    data = np.append(data, np.zeros([surplus_size, data.shape[1]]), axis=0)
    dataset_size = len(data) / sequence_length
    inputs = data[:-1].reshape(dataset_size, sequence_length, data.shape[1])
    labels = data[1:].reshape(dataset_size, sequence_length, data.shape[1])
    return inputs, labels


def create_midi_dataset(filename, sequence_length):
    midi = midi_converter.load_file(filename)
    dataset = midi_converter.convert_PrettyMIDI_to_train_data(midi)
    return [_div_inputs_and_label(one_data, sequence_length) for one_data in dataset]

if __name__ == '__main__':
    load_filename = "/home/tatsuya/Music/1.wav"
    dataset = create_dataset([load_filename], 20, 8000)
    inputs, labels = dataset[0]
    inputs = np.absolute(inputs)
    print inputs.max()
    print inputs.min()
    # print max(max(max(inputs)))
    # print min(min(min(inputs)))
