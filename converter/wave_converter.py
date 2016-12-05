import struct

import numpy as np
import wave
from itertools import chain


def load_wave(filename):
    wf = wave.open(filename, 'r')
    params = wf.getparams()
    chunk_size = wf.getnframes()
    amp = (2 ** 8) ** wf.getsampwidth() / 2
    data = wf.readframes(chunk_size)
    data = np.frombuffer(data, 'int16')
    data = data.astype(float)
    data = data / amp
    wf.close()

    return data, params


def fourier_transform(data, sampling_rate):
    data_size = len(data)
    result = [np.fft.fft(data[from_:from_ + sampling_rate])
              for from_ in range(0, data_size, sampling_rate)]
    return result


def inverse_fourier_transform(data):
    result = [np.fft.ifft(one_sec_data)
              for one_sec_data in data]
    return list(chain.from_iterable(result))


def write_wave(filename, params, data):
    data = np.array(data)
    amp = (2 ** 8) ** params[1] / 2
    data *= amp
    data = struct.pack("h" * len(data), *data)
    writer = wave.Wave_write(filename)
    writer.setparams(params)
    writer.writeframes(data)
    writer.close()


if __name__ == '__main__':
    load_filename = "/home/tatsuya/Music/2.wav"
    data, params = load_wave(load_filename)
    data = fourier_transform(data, params[2])
    a = [(max(i), min(i)) for i in data]
    data = inverse_fourier_transform(data)
    write_filename = "/home/tatsuya/Music/test.wav"
    write_wave(write_filename, params, data)
