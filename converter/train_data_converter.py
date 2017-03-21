# coding=utf-8
import numpy as np
import pretty_midi

import complex_converter
import midi_converter
import wave_converter


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


def generate_midi(filename, data, bar_size=32):
    '''
    0.5以上のものを1それ以下のものを0に変え、そのデータからmidiを作成する。
    :param filename:
    :param data:
    :return: None
    '''
    data = map(lambda one_data: map(lambda x: 1 if x > 0.5 else 0, one_data), data)
    midi_converter.save_file(filename, data, bar_size=bar_size)


def create_midi_train_data_set_v2(file_name_list, sequence_length, pitch_size, bar_size):
    train_data = []
    for file_name in file_name_list:
        try:
            train_data.append(_div_inputs_and_label(_create_midi_train_data_v2(file_name, pitch_size, bar_size), sequence_length))
        except:
            pass
    return train_data

def create_midi_train_data_set_with_file_name_v2(file_name_list, pitch_size, bar_size):
    train_data = []
    success_file_names = []
    for file_name in file_name_list:
        try:
            train_data.append(_create_midi_train_data_v2(file_name, pitch_size, bar_size))
            success_file_names.append(file_name)
        except:
            pass
    return train_data, success_file_names


def _midi_program_valid(midi):
    program_set = set()
    for instrument in midi.instruments:
        program_set.add(instrument.program)
    return len(program_set) == 1


def _create_midi_train_data_v2(file_name, pitch_size, bar_size):
    midi = midi_converter.load_file(file_name)
    if not _midi_program_valid(midi):
        raise Exception("has many instruments")

    tempo_change_times, tempi = midi.get_tempo_changes()
    if len(tempo_change_times) > 1:
        raise Exception("change tempo in music")

    one_data_sec = 60 / tempi[0] / bar_size * 4
    notes_map = {}
    for instrument in midi.instruments:
        for note in instrument.notes:
            start_time = int(round(note.start / one_data_sec))
            end_time = int(round(note.end / one_data_sec))
            if start_time not in notes_map:
                notes_map[start_time] = (end_time, [])
            max_end_time, notes = notes_map[start_time]
            notes.append(note)
            notes_map[start_time] = (max(max_end_time, end_time), notes)

    sorted_notes = sorted(notes_map.items(), key=lambda x: x[0])
    train_data = []
    for i, (start_time, (end_time, notes)) in enumerate(sorted_notes):
        if i != 0 and sorted_notes[i - 1][1][0] < start_time:
            pitch_data = [0] * pitch_size
            bar_data = [0] * bar_size
            bar_index = min([bar_size - 1, start_time - sorted_notes[i - 1][1][0]])
            bar_data[bar_index] = 1
            train_data.append(pitch_data + bar_data)

        pitch_data = [0] * pitch_size
        for note in notes:
            pitch_data[note.pitch] = 1

        bar_data = [0] * bar_size
        if i < len(sorted_notes) - 1:
            end_time = min([end_time, sorted_notes[i + 1][0]])
        bar_index = min([end_time - start_time - 1, bar_size - 1])
        bar_data[bar_index] = 1

        train_data.append(pitch_data + bar_data)
    return train_data


def generate_midi_v2(file_name, data, pitch_size, bar_size, velocity=100, instrument=0, tempo=120):
    if isinstance(instrument, int):
        instrument = pretty_midi.Instrument(program=instrument)
    elif isinstance(instrument, str):
        instrument = pretty_midi.instrument_name_to_program(instrument)

    one_data_sec = 60.0 / tempo / bar_size * 4.0
    current_time = 0.0
    for one_data in data:
        bar = np.argmax(one_data[pitch_size:]) + 1
        bar_time = bar * one_data_sec
        start_time = current_time
        end_time = current_time + bar_time
        for pitch, trigger in enumerate(one_data[:pitch_size]):
            if trigger == 1:
                note = pretty_midi.Note(velocity, pitch, start_time, end_time)
                instrument.notes.append(note)
        current_time += bar_time
    midi_converter.save_file_v2(file_name, [instrument], tempo)


if __name__ == '__main__':
    load_filename = "/home/tatsuya/Music/1.wav"
    dataset = create_dataset([load_filename], 20, 8000)
    inputs, labels = dataset[0]
    inputs = np.absolute(inputs)
    print inputs.max()
    print inputs.min()
    # print max(max(max(inputs)))
    # print min(min(min(inputs)))
