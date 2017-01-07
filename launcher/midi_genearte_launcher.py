import sys

import tensorflow as tf

sys.path.append("../")

# from launcher.common.generate import generate
from converter import train_data_converter
from model import LSTM


def generate(session, model, start_inputs, size, threshold_value):
    state = session.run(model.initial_state)
    fetches = [model.logits, model.last_state]
    result = [start_inputs[0]]
    logits = []
    start_input_size = len(start_inputs)

    for input_ in start_inputs:
        feed_dict = {model.inputs: [input_], model.initial_state: state}
        logits, state = session.run(fetches, feed_dict)
        logits = map(lambda x: 1 if x > threshold_value else 0, logits[0])
        result.append(logits)

    for i in range(size - start_input_size):
        feed_dict = {model.inputs: [[logits]], model.initial_state: state}
        logits, state = session.run(fetches, feed_dict)
        logits = map(lambda x: 1 if x > threshold_value else 0, logits[0])
        result.append(logits)
    return result


class Config:
    batch_size = 1
    sequence_length = 1
    input_size = 128
    label_size = 128
    cell_size_list = [128, 128]
    keep_prob = 1.0
    optimizer_function = tf.train.GradientDescentOptimizer(0.1)
    clip_norm = 3


if __name__ == '__main__':
    logdir = "../train_log/issue45/zanarukando/1/"
    with tf.Graph().as_default() as graph:
        config = Config
        load_filename = "../midi/train_zanarukandonite.midi"
        dataset = train_data_converter.create_midi_dataset(load_filename, config.sequence_length)
        inputs, labels = dataset[0]

        with tf.variable_scope("model"):
            model = LSTM.LSTM(True, config, True, tf.nn.sigmoid)

        with tf.Session() as session:
            saver = tf.train.Saver()
            session.run(tf.global_variables_initializer())
            saver.restore(session, logdir + "/data/model")
            buf = generate(session, model, inputs, len(inputs), 0.5)

        save_filename = "test.mid"
        train_data_converter.generate_midi(save_filename, buf)
