# coding=utf-8
import numpy as np
import complex_converter, wave_converter


def create_dataset(filename_list, sequence_length, sampling_rate):
    return [create_data(filename, sequence_length, sampling_rate) for filename in filename_list]


def create_data(filename, sequence_length, sampling_rate):
    data, params = wave_converter.load_wave(filename)
    data = data.astype(float)
    data /= 255
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

if __name__ == '__main__':
    load_filename = "/home/tatsuya/Music/1.wav"
    dataset = create_dataset([load_filename], 20, 8000)
    inputs, labels = dataset[0]
    inputs = np.absolute(inputs)
    print inputs.max()
    print inputs.min()
    # print max(max(max(inputs)))
    # print min(min(min(inputs)))
