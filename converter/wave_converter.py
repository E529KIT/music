import struct

import numpy as np
from scipy.io.wavfile import read, write

from itertools import chain


def load_wave(filename):
    sampling_rate, data = read(filename)
    data = data.astype(float)
    data /= 255
    return data, sampling_rate


def fourier_transform(data, sampling_rate):
    data_size = len(data)
    result = [np.fft.fft(data[from_:from_ + sampling_rate])
              for from_ in range(0, data_size, sampling_rate)]
    return result


def inverse_fourier_transform(data):
    result = [np.fft.ifft(one_sec_data)
              for one_sec_data in data]
    return list(chain.from_iterable(result))


def write_wave(filename, sampling_rate, data):
    data = np.array(data)
    data *= 255
    data = data.astype('uint8')
    write(filename, sampling_rate, data)


if __name__ == '__main__':
    load_filename = "/home/tatsuya/Music/1.wav"
    data, sampling_rate = load_wave(load_filename)
    data = fourier_transform(data, sampling_rate)
    data = inverse_fourier_transform(data)
    write_filename = "/home/tatsuya/Music/test.wav"
    write_wave(write_filename, sampling_rate, data)
