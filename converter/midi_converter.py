# coding=utf-8
import pretty_midi
import numpy as np


def load_file(filename):
    return pretty_midi.PrettyMIDI(filename)


# todo: 途中でテンポの変わる曲にも対応できると良い
# todo: 音の大きさの変化も学習できるといいかも
def convert_PrettyMIDI_to_train_data(midi):
    if not isinstance(midi, pretty_midi.PrettyMIDI):
        raise Exception('arg != PrettyMIDI')
    end_time = midi.get_end_time()
    tempo_change_times, tempi = midi.get_tempo_changes()
    if len(tempo_change_times) > 1:
        raise Exception("change tempo in music")
    one_data_sec = 60 / tempi[0] / 8
    train_data_size = end_time / one_data_sec

    # 同じ楽器で演奏されているものはまとめる。
    instruments = [instrument for instrument in midi.instruments if not instrument.is_drum]
    programs = {}
    program_index = 0
    for instrument in instruments:
        if programs.has_key(instrument.program):
            continue
        programs[instrument.program] = program_index
        program_index += 1

    train_data = np.zeros([len(programs), train_data_size, 128])
    for instrument in instruments:
        train_data_index = programs[instrument.program]
        for note in instrument.notes:
            start_time = int(note.start / one_data_sec)
            end_time = int(note.end / one_data_sec)
            for time in range(start_time, end_time):
                train_data[train_data_index][time][note.pitch] = 1
    return train_data


def save_file(filename, data, velocity=100, instrument=pretty_midi.Instrument(program=0), tempo=120):
    if isinstance(instrument, pretty_midi.Instrument):
        instrument = instrument
    elif isinstance(instrument, str):
        instrument = pretty_midi.instrument_name_to_program(instrument)
    else:
        raise Exception('instrument error')

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    one_data_sec = 60.0 / tempo / 8.0
    # 音のなり始めるインデッックスを覚えておくためのもの
    start_time_list = np.zeros(128)
    max_time = 0
    for i, pitches in enumerate(data):
        for pitch, on_off in enumerate(pitches):
            # todo: データの閾値等どうするか
            # 今のバージョンではconvertのデータに対応できるように作っている。
            if on_off == 1:
                if start_time_list[pitch] == 0:
                    start_time_list[pitch] = i
            else:
                if start_time_list[pitch] != 0:
                    start_time = start_time_list[pitch] * one_data_sec
                    end_time = i * one_data_sec
                    max_time = max(max_time, end_time)
                    note = pretty_midi.Note(velocity, pitch, start_time, end_time)
                    instrument.notes.append(note)
                    start_time_list[pitch] = 0
    midi.instruments.append(instrument)
    midi.write(filename)


if __name__ == '__main__':
    load_filename = "../midi/bwv772.midi"
    save_filename = "test.mid"
    midi = load_file(load_filename)
    train_data = convert_PrettyMIDI_to_train_data(midi)
    save_file(save_filename, train_data[0])
