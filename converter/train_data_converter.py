# coding=utf-8
import complex_converter, wave_converter

def create_dataset(filename_list, sequence_length, sampling_rate):
    return [create_data(filename, sequence_length, sampling_rate) for filename in filename_list]

def create_data(filename, sequence_length, sampling_rate):
    data, params = wave_converter.load_wave(filename)
    data = wave_converter.fourier_transform(data, sampling_rate)
    if len(data[-1]) != sampling_rate:
        data.pop()  # 音楽の時間が小数点秒の場合、データを食わせられないため
    data = [complex_converter.complex_list_to_double_list(one_data) for one_data in data]
    data_size = len(data)
    inputs = [data[i:i + sequence_length]
              for i in range(0, data_size, sequence_length)]
    labels = [data[i: i + sequence_length]
              for i in range(1, data_size, sequence_length)]
    # 余剰データを0でうめる
    surplus_size = sequence_length - (data_size % sequence_length)
    zero_list = [0] * surplus_size
    inputs[-1] = inputs[-1] + zero_list
    labels[-1] = labels[-1] + zero_list
    return inputs, labels

if __name__ == '__main__':
    load_filename = "/Users/tatsuya/Music/0321.wav"
    dataset = create_dataset([load_filename], 100, 44100)
    print dataset