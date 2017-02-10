# coding=utf-8
import pretty_midi
import numpy as np


def load_file(filename):
    return pretty_midi.PrettyMIDI(filename)


# todo: 途中でテンポの変わる曲にも対応できると良い
# todo: 音の大きさの変化も学習できるといいかも
def convert_PrettyMIDI_to_train_data(midi, pitch_sequence=True, bar_size=32):
    '''
    midiから訓練データを作成する。
    ここで生成される訓練データは、一生節あたりのsequence_lengthがbar_sizeで、one_hotみたいな形で[音のなっている時間][音の高さ] = 1となっている。
    :param midi: pretty_midi
    :param pitch_sequence: True or False.
    Trueのとき、一つの音の開始から終了まで１で埋める。
    Falseのとき、音のなり始めのところのみを1とする。
    :param bar_size: 一生節あたりの訓練データのsequenceサイズ
    :return: [len(instruments), train_data_sequence, 128]
    '''
    if not isinstance(midi, pretty_midi.PrettyMIDI):
        raise Exception('arg != PrettyMIDI')
    end_time = midi.get_end_time()
    tempo_change_times, tempi = midi.get_tempo_changes()
    if len(tempo_change_times) > 1:
        raise Exception("change tempo in music")
    one_data_sec = 60 / tempi[0] / bar_size * 4
    train_data_size = int(round(end_time / one_data_sec))

    # 同じ楽器で演奏されているものはまとめる。
    # ドラムは学習データから外す
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
            start_time = int(round(note.start / one_data_sec))
            end_time = int(round(note.end / one_data_sec))
            if pitch_sequence:
                for time in range(start_time, end_time):
                    train_data[train_data_index][time][note.pitch] = 1
            else:
                train_data[train_data_index][start_time][note.pitch] = 1
    return train_data


def save_file(filename, data, velocity=100, instrument=pretty_midi.Instrument(program=0), tempo=120, bar_size=32):
    if isinstance(instrument, pretty_midi.Instrument):
        instrument = instrument
    elif isinstance(instrument, str):
        instrument = pretty_midi.instrument_name_to_program(instrument)
    else:
        raise Exception('instrument error')

    # pretty_midi.Instrument(program=0)でInstrumentを生成した場合、初期化されずに前のデータが残っている
    instrument.notes = []

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    one_data_sec = 60.0 / tempo / bar_size * 4.0
    # 音のなり始めるインデッックスを覚えておくためのもの
    start_time_list = np.zeros(128)
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
                    note = pretty_midi.Note(velocity, pitch, start_time, end_time)
                    instrument.notes.append(note)
                    start_time_list[pitch] = 0
    midi.instruments.append(instrument)
    midi.write(filename)


def save_file_v2(filename, instruments, tempo=120):
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    midi.instruments.extend(instruments)
    midi.write(filename)

if __name__ == '__main__':
    load_filename = "../midi/zanarukandonite.midi"
    save_filename = "test.mid"
    midi = load_file(load_filename)
    train_data = convert_PrettyMIDI_to_train_data(midi)
    save_file(save_filename, train_data[0])
