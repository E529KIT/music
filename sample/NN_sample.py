# coding=utf-8
import tensorflow as tf
import numpy as np

from model.NN import NN


class Config(object):
    input_size = 1
    label_size = 3
    hidden_layer = [20, 30, 40]
    keep_prob = 1
    optimizer_function = tf.train.GradientDescentOptimizer(0.1)


config = Config
with tf.variable_scope("model"):
    model = NN(True, config)
with tf.variable_scope("model", reuse=True):
    eval_model = NN(False, config)

# 入力値の番地を１にする
# 下記の出力になるようにtrainする
#  [0] -> [1, 0, 0]
#  [1] -> [0, 1, 0]
#  [2] -> [0, 0, 1]
with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    # trainデータ作成
    inputs = []
    labels = []
    for i in range(3):
        input_ = [i]
        label = np.zeros(3)
        label[i] = 1
        inputs.append(input_)
        labels.append(label)

    # train
    for epoch in range(100):
        _, loss = session.run([model.traion_op, model.loss], {model.inputs: inputs, model.labels: labels})
        if epoch % 10 == 0:
            print loss

    # eval
    correct_prediction = tf.equal(tf.argmax(eval_model.labels, 1), tf.argmax(eval_model.logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_rate = session.run(accuracy, {eval_model.inputs: inputs, eval_model.labels: labels})
    print accuracy_rate
