# coding=utf-8
'''
LSTM層をpitchとbarで分ける
'''
import tensorflow as tf


def _create_multi_lstm_cell(inputs, cell_size_list, batch_size, scope, keep_prob=None):
    cells = []
    for cell_size in cell_size_list:
        cell = tf.contrib.rnn.BasicLSTMCell(cell_size, forget_bias=0)
        if keep_prob and keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        cells.append(cell)
    cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, parallel_iterations=1,
                                       swap_memory=True,
                                       scope=scope)
    return initial_state, outputs, state


class Model:
    def __init__(self, config, inputs=None, labels=None, lengths=None, generate=False):
        self.batch_size = batch_size = config.batch_size
        pitch_size = config.pitch_size
        bar_size = config.bar_size
        input_size = pitch_size + bar_size
        label_size = input_size

        self._global_step = global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)

        with tf.name_scope("interface"):
            if inputs is None:
                inputs = tf.placeholder(tf.float32, [batch_size, config.sequence_length, input_size],
                                        "input_data")
            self._inputs = inputs
            self._pitch_inputs = pitch_inputs = tf.slice(inputs, [0, 0, 0], [-1, -1, pitch_size])
            self._bar_inputs = bar_inputs = tf.slice(inputs, [0, 0, pitch_size], [-1, -1, bar_size])

            if labels is None:
                labels = tf.placeholder(tf.float32, [batch_size, config.sequence_length, label_size], "labels")
            self._labels = labels
            self._pitch_labels = pitch_labels = tf.slice(labels, [0, 0, 0], [-1, -1, pitch_size])
            self._bar_labels = bar_labels = tf.slice(labels, [0, 0, pitch_size], [-1, -1, bar_size])

        with tf.name_scope("hidden_layer"):
            with tf.name_scope("pitch"):
                with tf.name_scope("cnn") as scope:
                    filter_width = 12
                    cnn_out_size = config.cnn_out_size
                    outputs = []
                    for i in range(pitch_size - filter_width):
                        reuse = i == 0
                        cnn_in = tf.slice(pitch_inputs, [0, 0, i], [-1, -1, filter_width])
                        output = tf.contrib.layers.fully_connected(inputs=cnn_in, num_outputs=cnn_out_size,
                                                                   scope=scope, activation_fn=tf.nn.sigmoid,
                                                                   reuse=reuse)
                        outputs.append(output)
                    cnn_out = tf.concat(outputs, 2)

                with tf.name_scope("LSTM") as scope:
                    mix_inputs = tf.concat([cnn_out, bar_inputs], 2)
                    pitch_init_state, pitch_outputs, pitch_last_state \
                        = _create_multi_lstm_cell(mix_inputs, config.pitch_cell_size_list,
                                                  batch_size, scope, config.keep_prob)
                    self._pitch_init_state = pitch_init_state
                    self._pitch_last_state = pitch_last_state

            with tf.name_scope("bar"):
                with tf.name_scope("LSTM") as scope:
                    bar_init_state, bar_outputs, bar_last_state \
                        = _create_multi_lstm_cell(inputs, config.bar_cell_size_list,
                                                  batch_size, scope, config.keep_prob)
                    self._bar_init_state = bar_init_state
                    self._bar_last_state = bar_last_state

            with tf.name_scope("last_layer"):
                concat_pitch_bar_lstm_out = tf.concat([pitch_outputs, bar_outputs], 2)
                with tf.name_scope("pitch") as scope:
                    self._pitch_logits = pitch_logits = \
                        tf.contrib.layers.fully_connected(inputs=concat_pitch_bar_lstm_out,
                                                          num_outputs=config.pitch_size,
                                                          scope=scope, activation_fn=tf.nn.sigmoid)
                with tf.name_scope("bar") as scope:
                    self._bar_logits = bar_logits \
                        = tf.contrib.layers.fully_connected(inputs=concat_pitch_bar_lstm_out,
                                                            num_outputs=config.bar_size,
                                                            scope=scope, activation_fn=tf.nn.softmax)
            self._init_state = tuple([pitch_init_state, bar_init_state])
            self._last_state = tuple([pitch_last_state, bar_last_state])

        if not generate:
            with tf.name_scope("loss"):
                mask_flat = tf.reshape(tf.sequence_mask(lengths, dtype=tf.float32), [-1])

                with tf.name_scope("pitch_los"):
                    pitch_labels_flat = tf.reshape(pitch_labels, [-1, pitch_size])
                    pitch_logits_flat = tf.reshape(pitch_logits, [-1, pitch_size])
                    # 出力がすべて０になってしまっているので、labelが1のところのlossのweightを高くしてみる。
                    loss_weight = tf.clip_by_value(pitch_labels_flat * config.on_pitch_loss_weight, 1,
                                                   config.on_pitch_loss_weight)
                    pitch_loss = tf.reduce_mean(loss_weight * tf.square(pitch_logits_flat - pitch_labels_flat), 1)
                    self._pitch_loss = pitch_loss = tf.reduce_mean(mask_flat * pitch_loss)
                    tf.summary.scalar('pitch_loss', pitch_loss)

                with tf.name_scope("bar_loss"):
                    bar_labels_flat = tf.reshape(bar_labels, [-1, bar_size])
                    bar_logits_flat = tf.reshape(bar_logits, [-1, bar_size])
                    bar_loss = tf.reduce_mean(-bar_labels_flat * tf.log(tf.clip_by_value(bar_logits_flat, 1e-10, 1.0)),
                                              1)
                    self._bar_loss = bar_loss = tf.reduce_mean(mask_flat * bar_loss)
                    tf.summary.scalar('bar_loss', bar_loss)

                self._logits = tf.concat([pitch_logits, bar_logits], 2)

                self._loss = loss = pitch_loss + bar_loss
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
                    tf.summary.scalar('gradient/mean', tf.reduce_mean(abs_gradient))

    @property
    def inputs(self):
        return self._inputs

    @property
    def pitch_inputs(self):
        return self._pitch_inputs

    @property
    def bar_inputs(self):
        return self._bar_inputs

    @property
    def labels(self):
        return self._labels

    @property
    def pitch_labels(self):
        return self._pitch_labels

    @property
    def bar_labels(self):
        return self._bar_labels

    @property
    def initial_state(self):
        return self._init_state

    @property
    def pitch_initial_state(self):
        return self._pitch_init_state

    @property
    def bar_initial_state(self):
        return self._bar_init_state

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
    def pitch_last_state(self):
        return self._pitch_last_state

    @property
    def bar_last_state(self):
        return self._bar_last_state

    @property
    def pitch_logits(self):
        return self._pitch_logits

    @property
    def bar_logits(self):
        return self._bar_logits

    @property
    def logits(self):
        return self._logits
