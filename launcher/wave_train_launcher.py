# coding=utf-8
import tensorflow as tf
import sys
import datetime

sys.path.append("../")

from converter import train_data_converter
from model import LSTM


def train(session, model, inputs, labels, epoch, saver):
    train_writer = tf.train.SummaryWriter('./log', session.graph)
    merged = tf.merge_all_summaries()
    session.run(tf.global_variables_initializer())
    global_step, state = session.run([model.global_step, model.initial_state])
    fetches = [model.train_optimizer, model.global_step, model.last_state, model.loss, merged]
    for i in range(epoch):
        for input_, label in zip(inputs, labels):
            feed_dict = {model.inputs: [input_], model.initial_state: state, model.labels: [label]}
            _, global_step, state, loss, summaries = session.run(fetches, feed_dict)
            train_writer.add_summary(summaries, global_step)
        if i % 50 == 0:
            saver.save(session, "data/1/model")
            now = datetime.datetime.today()
            print '%d: %s' % (i, now.isoformat())



class Config:
    batch_size = 1
    sequence_length = 20
    # 8KHz / 2 * 2 (sampling rate / 2 * complex)
    input_size = 8000
    label_size = 8000
    cell_size_list = [4000]
    keep_prob = 1.0
    optimizer_function = tf.train.AdamOptimizer(0.1)


if __name__ == '__main__':
    with tf.Graph().as_default() as graph:
        config = Config

        load_filename = "/home/tatsuya/Music/1.wav"
        dataset = train_data_converter.create_dataset([load_filename], config.sequence_length, 8000)
        inputs, labels = dataset[0]

        with tf.variable_scope("model"):
            model = LSTM.LSTM(True, config, last_activation_function=tf.nn.tanh)

        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, "data/1/model")
            train(session, model, inputs, labels, 1000, saver)
