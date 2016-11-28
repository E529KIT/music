# coding=utf-8
# このコードは学習済みデータの一部転用と一部のレイヤーのみの学習のテストコードとしてつかう

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# Create the model
def variable_summaries(var, name):
    """varの値の変動をtensorboardで確認できるようにする"""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('stddev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


def create_layer(input_tensor, input_dim, output_dim, layer_name, activation_function=tf.nn.relu):
    """
    input_tensorに新しいoutput_dim数のレイヤー付け加える
    :param input_tensor:
    :param input_dim:
    :param output_dim:
    :param layer_name:
    :param activation_function:
    :return: 付け加えたあとのtensor, weighs, biases
    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.truncated_normal([output_dim], stddev=0.1))
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.histogram_summary(layer_name + '/pre_activations', preactivate)
        activations = activation_function(preactivate)
        tf.histogram_summary(layer_name + '/activations', activations)
        return activations, weights, biases


x = tf.placeholder(tf.float32, [None, 784])
logits, weights1, biases1 = create_layer(x, 784, 784, "layer1")
logits, weights2, biases2 = create_layer(logits, 784, 10, "layer2", activation_function=tf.nn.softmax)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_))
tf.scalar_summary('loss', cross_entropy)
train_opt = tf.train.GradientDescentOptimizer(0.5)
update_params = [weights2, biases2]
# もうひとつのやり方
# 下記のように、学習対象としない変数を trainable=Falseと設定することで tf.trainable_variables() でtrainable=Trueになっているものだけを取得できる。
# weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1), trainable=False)
# update_params = tf.trainable_variables()
gradients = tf.gradients(cross_entropy, update_params)
train_step = train_opt.apply_gradients(zip(gradients,update_params))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

# For Tensor Board
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter('./log', sess.graph)

# Train
for step in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    feed_dict = {x: batch_xs, y_: batch_ys}
    if step % 10 == 0:
        # record weight and biases
        _, summaries = sess.run([train_step, merged], feed_dict)
        train_writer.add_summary(summaries, step)
    else:
        sess.run(train_step, feed_dict)

# Test trained model
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))
