# coding=utf-8
'''
音楽の性質を取り入れたモデル
コードの表現をモデルに取り入れる。
コードは音程12、種類（マイナー,7等）14種類あるのでそれを表現する
大量の音楽で学習させれば、転移学習に用いれる可能性あり
'''
import tensorflow as tf


class DefaultConfig:
    batch_size = 5
    sequence_length = 36
    pitch_size = 128
    bar_size = 32
    cell_size_list = [20, 30, 40]
    keep_prob = 0.9
    optimizer_function = tf.train.AdamOptimizer(0.5)
    clip_norm = 3
    pitch_loss_wight = 1
    cnn_out_size = 1


class Model:
    def __init__(self, config=DefaultConfig, generate=False, inputs=None, labels=None, activate_function=tf.nn.relu):
        self.batch_size = batch_size = config.batch_size
        self.sequence_length = sequence_length = config.sequence_length
        pitch_size = config.pitch_size
        bar_size = config.bar_size
        input_size = pitch_size + bar_size
        label_size = input_size

        self._global_step = global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)

        with tf.name_scope("interface"):
            if generate:
                self._inputs = inputs = tf.placeholder(tf.float32, [batch_size, sequence_length, input_size],
                                                       "generate_input")
            else:
                if inputs is None:
                    inputs = tf.placeholder(tf.float32, [batch_size, sequence_length, input_size],
                                            "input_data")
                self._inputs = inputs
            if labels is None:
                labels = tf.placeholder(tf.float32, [batch_size, sequence_length, label_size], "labels")
            self._labels = labels

        with tf.name_scope("hidden_layer") as scope:
            with tf.name_scope("cnn") as cnn_scope:
                filter_width = 12
                #  number of code types
                cnn_out_size = 14
                pitch_inputs = tf.slice(inputs, [0, 0, 0], [-1, -1, pitch_size])
                pitch_inputs_flat = tf.reshape(pitch_inputs, [-1, pitch_size, 1])
                with tf.variable_scope(cnn_scope):
                    cnn_w = tf.get_variable("weight", [filter_width, 1, cnn_out_size], tf.float32)
                    cnn_b = tf.get_variable("bias", [cnn_out_size], tf.float32)
                cnn_out = tf.nn.conv1d(pitch_inputs_flat, cnn_w, 1, 'SAME') + cnn_b
                cnn_out = activate_function(cnn_out + cnn_b)
                cnn_out = tf.reshape(cnn_out, [batch_size, sequence_length, -1])
                bar_input = tf.slice(inputs, [0, 0, pitch_size], [-1, -1, bar_size])
                cnn_out = tf.concat([cnn_out, bar_input], 2)

            with tf.name_scope("LSTM"):
                cells = []
                for cell_size in config.cell_size_list:
                    cell = tf.contrib.rnn.BasicLSTMCell(cell_size, forget_bias=0, activation=activate_function)
                    if config.keep_prob < 1:
                        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
                    cells.append(cell)
                cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
                self._initial_state = cell.zero_state(batch_size, dtype=tf.float32)
                outputs, state = tf.nn.dynamic_rnn(cell, cnn_out, [sequence_length] * batch_size, self._initial_state,
                                                   parallel_iterations=1, swap_memory=True, scope=scope)
                self._last_state = state

            with tf.name_scope("last_layer") as scope:
                logits = tf.contrib.layers.fully_connected(inputs=outputs, num_outputs=label_size, scope=scope)

        # split pitch and bar to calc loss
        with tf.name_scope("loss"):
            bar_labels = tf.slice(labels, [0, 0, pitch_size], [-1, -1, bar_size])
            bar_labels_flat = tf.reshape(bar_labels, [-1, bar_size])
            bar_logits = tf.slice(logits, [0, 0, pitch_size], [-1, -1, bar_size])
            self._bar_logits = bar_logits = tf.nn.softmax(bar_logits)
            bar_logits_flat = tf.reshape(bar_logits, [-1, bar_size])
            self._bar_loss = bar_loss = tf.reduce_mean(
                -bar_labels_flat * tf.log(tf.clip_by_value(bar_logits_flat, 1e-10, 1.0)), name="bar_loss")
            tf.summary.scalar('bar_loss', bar_loss)

            pitch_labels = tf.slice(labels, [0, 0, 0], [-1, -1, pitch_size])
            pitch_labels_flat = tf.reshape(pitch_labels, [-1, pitch_size])
            pitch_logits = tf.slice(logits, [0, 0, 0], [-1, -1, pitch_size])
            self._pitch_logits = pitch_logits = tf.clip_by_value(pitch_logits, 0, 1, "pitch_logits")
            pitch_logits_flat = tf.reshape(pitch_logits, [-1, pitch_size])
            # 全部0のデータ（余剰データ）はlossの値に含めない
            trigger_loss = tf.reshape(tf.reduce_max(bar_labels_flat, axis=1), [-1, 1])
            self._pitch_loss = pitch_loss = tf.reduce_mean(
                trigger_loss * tf.pow(pitch_logits_flat - pitch_labels_flat, 4),
                name="pitch_loss")
            tf.summary.scalar('pitch_loss', pitch_loss)

            self._logits = tf.concat([pitch_logits, bar_logits], 2)

            self._loss = loss = config.pitch_loss_wight * pitch_loss + bar_loss
            tf.summary.scalar('loss', loss)

        with tf.name_scope("train"):
            params = tf.trainable_variables()
            gradients = tf.gradients(loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients,
                                                          config.clip_norm)
            self._train_optimizer = config.optimizer_function.apply_gradients(zip(clipped_gradients, params),
                                                                              global_step)

        for param, gradient in zip(params, clipped_gradients):
            abs_gradient = tf.abs(gradient)
            # not valid contain ":"
            # variable name is "....:0"
            # TODO: Every name support. This is not fulfill such as "....:10".
            with tf.name_scope(param.name[:-2]):
                # tf.summary.scalar('gradient/max', tf.reduce_max(abs_gradient))
                # tf.summary.scalar('gradient/min', tf.reduce_min(abs_gradient))
                tf.summary.scalar('gradient/mean', tf.reduce_mean(abs_gradient))

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def bar_loss(self):
        return self._bar_loss

    @property
    def pitch_loss(self):
        return self._pitch_loss

    @property
    def loss(self):
        return self._loss

    @property
    def global_step(self):
        return self._global_step

    @property
    def train_optimizer(self):
        return self._train_optimizer

    @property
    def last_state(self):
        return self._last_state

    @property
    def pitch_logits(self):
        return self._pitch_logits

    @property
    def bar_logits(self):
        return self._bar_logits

    @property
    def logits(self):
        return self._logits
