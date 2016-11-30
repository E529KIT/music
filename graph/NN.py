# coding=utf-8
import tensorflow as tf


class DefaultConfig:
    input_size = 20
    label_size = 20
    hidden_layer = [20, 30, 40]
    keep_prob = 0.9
    optimizer_function = tf.train.AdamOptimizer(0.5)


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


def create_layer(input_tensor, input_dim, output_dim, layer_name, activation_function=None):
    """
    input_tensorに新しいoutput_dim数のレイヤー付け加える
    :param input_tensor:入力のtensor
    :param input_dim:入力tensorのサイズ
    :param output_dim:レイヤー数
    :param layer_name:レイヤーの名前
    :param activation_function:活性化関。Noneの場合はなし
    :return: 付け加えたあとのtensor
    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.get_variable('weights', [input_dim, output_dim])
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = tf.get_variable('biases', [output_dim])
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.histogram_summary(layer_name + '/pre_activations', preactivate)
        if activation_function is not None:
            preactivate = activation_function(preactivate)
        tf.histogram_summary(layer_name + '/activations', preactivate)
        return preactivate, weights, biases


class NN:
    def __init__(self, is_train, config=DefaultConfig, activation_function=tf.nn.relu):
        self._global_step = global_step = tf.Variable(0, False, dtype=tf.int16)

        self._inputs = inputs = tf.placeholder(tf.float32, [None, config.input_size])
        self._labels = labels = tf.placeholder(tf.float32, [None, config.label_size])

        self._weights_list = weights_list = []
        self._biases_list = biases_list = []
        logits_dim = config.input_size
        logits = inputs
        for i, hidden_size in enumerate(config.hidden_layer):
            layer_name = "layer%d" % (i)
            with tf.variable_scope(layer_name):
                logits, weights, biases = create_layer(logits, logits_dim, hidden_size, layer_name, activation_function)
                weights_list.append(weights)
                biases_list.append(biases)
                if is_train and config.keep_prob < 1:
                    logits = tf.nn.dropout(logits, config.keep_prob)
            logits_dim = hidden_size
        with tf.variable_scope("soft_max_layer"):
            logits, weights, biases = create_layer(logits, logits_dim, config.label_size, "soft_max_layer")
            self._logits = logits
            weights_list.append(weights)
            biases_list.append(biases)
        self._loss = loss = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
        self._train_op = config.optimizer_function.minimize(loss, global_step)

    @property
    def global_step(self):
        return self._global_step

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def weights_list(self):
        return self._weights_list

    @property
    def biases_list(self):
        return self.biases_list

    @property
    def logits(self):
        return self._logits

    @property
    def loss(self):
        return self._loss

    @property
    def traion_op(self):
        return self._train_op
