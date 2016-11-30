# coding=utf-8
import tensorflow as tf
from model.LSTM import LSTM
import random
import numpy as np

class Config:
    batch_size = 3
    sequence_length = 3
    input_size = 100
    label_size = 10
    cell_size_list = [100, 100, 100]
    keep_prob = 0.9
    optimizer_function = tf.train.AdamOptimizer(0.1)


config = Config
with tf.variable_scope("model"):
    model = LSTM(True, config)
with tf.variable_scope("model", reuse=True):
    eval_model = LSTM(False, config)

# ***WARNING: モデルが難しすぎて学習しない***
# 3つの中から最大の値を足していき、その合計を10で割ったあまり
# [[1, 2, 3], [2, 5, 0]]の場合
# [1, 2, 3]の最大値3と[2, 5, 0]の最大値5を足して3 + 5 = 8となり8%10=8が出力されるようなモデル

def create_data(length):
    inputs = []
    labels = []
    sum_ = 0
    for _ in range(length):
        input_ = [random.randint(1, 100) for _ in range(3)]
        sum_ += max(input_)
        label = np.zeros([10])
        label[sum_ % 10] = 1
        inputs.append([int(i in input_) for i in range(100)])
        labels.append([int(sum_ % 10 == i) for i in range(10)])
    return inputs, labels


def create_data_set(batch_size, length):
    inputs = []
    labels = []
    for _ in range(batch_size):
        i, l = create_data(length)
        inputs.append(i)
        labels.append(l)
    return inputs, labels


with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    # train
    for epoch in range(1000):
        inputs, labels = create_data_set(config.batch_size, config.sequence_length)
        _, loss = session.run([model.train_optimizer, model.loss],
                              feed_dict={model.inputs: inputs, model.labels: labels})
        if epoch % 100 == 0:
            print loss

    # eval
    correct_prediction = tf.equal(tf.argmax(eval_model.labels_flat, 1), tf.argmax(eval_model.logits_flat, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    for i in range(1, 10):
        loss_sum = 0.
        length = i * config.sequence_length
        inputs, labels = create_data_set(config.batch_size, length)
        state = session.run(model.initial_state)
        for step in range(i):
            start_index = step * config.sequence_length
            end_index = start_index + config.sequence_length
            inputs_one_step = [inputs[index][start_index:end_index][:] for index in range(config.sequence_length)]
            labels_one_step = [labels[index][start_index:end_index][:] for index in range(config.sequence_length)]
            loss, state = session.run([accuracy, eval_model.last_state],
                                      {eval_model.initial_state: state, eval_model.inputs: inputs_one_step,
                                       eval_model.labels: labels_one_step})
            loss_sum += loss
        print "%d: %f" % (i, loss_sum / i)
