import tensorflow as tf
import numpy as np

from converter import wave_converter
from model import LSTM

class Config:
    batch_size = 1
    sequence_length = 128
    input_size = 44100
    label_size = 44100
    cell_size_list = [100, 100, 100]
    keep_prob = 0.9
    optimizer_function = tf.train.AdamOptimizer(0.1)


config = Config
with tf.variable_scope("model"):
    model = LSTM.LSTM(True, config)
with tf.variable_scope("model", reuse=True):
    generate_model = LSTM.LSTM(True, config, True)

load_filename = "/home/tatsuya/Music/2.wav"
data, params = wave_converter.load_wave(load_filename)
data = wave_converter.fourier_transform(data, params[2])
data_size = len(data)
inputs = [data[i:i + config.sequence_length]
          for i in range(0, data_size, config.sequence_length)]
labels = [data[i: i + config.sequence_length]
          for i in range(1, data_size, config.sequence_length)]
inputs.pop()
labels.pop()


with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    global_step, state = session.run([model.global_step, model.initial_state])
    fetches = [model.train_optimizer, model.global_step, model.last_state]
    for input_, label in zip(inputs, labels):
        feed_dict = {model.inputs: [input_], model.initial_state: state, model.labels: [label]}
        _, global_step, state = session.run(fetches, feed_dict)

    state = session.run(generate_model.initial_state)
    fetches = [generate_model.logits, generate_model.last_state]
    buf = []
    logits = [inputs[0][0]]
    for i in range(data_size):
        feed_dict = {generate_model.inputs: [logits], generate_model.initial_state: state}
        logits, state = session.run(fetches, feed_dict)
        buf.append(data[0])

write_filename = "/home/tatsuya/Music/deep.wav"
buf = np.array(buf)
buf = wave_converter.inverse_fourier_transform(buf)
wave_converter.write_wave(write_filename, params, buf)