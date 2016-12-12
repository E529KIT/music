# coding=utf-8
import tensorflow as tf
import sys
sys.path.append("../")

from converter import complex_converter
from converter import train_data_converter
from converter import wave_converter
from model import LSTM


def generate(session, model, start_input, size):
    state = session.run(model.initial_state)
    fetches = [model.logits, model.last_state]
    result = []
    logits = [start_input]
    for i in range(size):
        feed_dict = {model.inputs: [logits], model.initial_state: state}
        logits, state = session.run(fetches, feed_dict)
        result.append(logits)
    return result


def train(session, model, inputs, labels, epoch):
    session.run(tf.initialize_all_variables())
    global_step, state = session.run([model.global_step, model.initial_state])
    fetches = [model.train_optimizer, model.global_step, model.last_state]
    for _ in range(epoch):
        for input_, label in zip(inputs, labels):
            feed_dict = {model.inputs: [input_], model.initial_state: state, model.labels: [label]}
            _, global_step, state = session.run(fetches, feed_dict)


class Config:
    batch_size = 1
    sequence_length = 128
    input_size = 44100 * 2  # 44.1KHz 複素数
    label_size = 44100 * 2
    cell_size_list = [100, 100, 100]
    keep_prob = 0.9
    optimizer_function = tf.train.AdamOptimizer(0.1)


if __name__ == '__main__':
    config = Config
    with tf.variable_scope("model"):
        model = LSTM.LSTM(True, config, last_activation_function=tf.nn.tanh)
    with tf.variable_scope("model", reuse=True):
        generate_model = LSTM.LSTM(True, config, True, last_activation_function=tf.nn.tanh)

    load_filename = "/Users/tatsuya/Music/0321.wav"
    dataset = train_data_converter.create_dataset([load_filename], config.sequence_length, 44100)
    inputs, labels = dataset[0]

    with tf.Session() as session:
        train(session, model, inputs, labels, 1)
        buf = generate(session, generate_model, inputs[0][0], 1)

    buf = complex_converter.double_list_to_complex_list(buf)
    write_filename = "/Users/tatsuya/Music/created.wav"
    buf = wave_converter.inverse_fourier_transform(buf)
    create_wave_params = (1, 2, 44100, len(buf), 'NONE', 'not compressed')
    wave_converter.write_wave(write_filename, create_wave_params, buf)
