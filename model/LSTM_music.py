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


class LSTM:
    def __init__(self, config=DefaultConfig, generate=False):
        self.batch_size = batch_size = config.batch_size
        self.sequence_length = sequence_length = config.sequence_length
        pitch_size = config.pitch_size
        bar_size = config.bar_size
        input_size = pitch_size + bar_size
        label_size = input_size

        self._global_step = global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)

        if generate:
            self._inputs = inputs = tf.placeholder(tf.float32, [batch_size, sequence_length, input_size],
                                                   "generate_input")
        else:
            self._inputs = inputs = tf.placeholder(tf.float32, [batch_size, sequence_length, input_size],
                                                   "input_data")
        self._labels = labels = tf.placeholder(tf.float32, [batch_size, sequence_length, label_size], "labels")

        cells = []
        for cell_size in config.cell_size_list:
            cell = tf.nn.rnn_cell.BasicLSTMCell(cell_size, forget_bias=0)
            if config.keep_prob < 1:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
            cells.append(cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        self._initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, [sequence_length] * batch_size, self._initial_state,
                                           parallel_iterations=1, swap_memory=True)
        self._last_state = state

        logits = tf.contrib.layers.fully_connected(inputs=outputs, num_outputs=label_size)

        # split pitch and bar to calc loss
        pitch_labels_flat = tf.slice(labels, [0, 0, 0], [-1, -1, pitch_size])
        pitch_logits = tf.slice(logits, [0, 0, 0], [-1, -1, pitch_size])
        self._pitch_logits = pitch_logits = tf.nn.sigmoid(pitch_logits, "pitch_logits")
        pitch_logits_flat = tf.reshape(pitch_logits, [-1, pitch_size])
        self._pitch_loss = pitch_loss = tf.reduce_mean(tf.square(pitch_logits_flat - pitch_labels_flat), name="pitch_loss")
        tf.summary.scalar('pitch_loss', pitch_loss)

        bar_labels_flat = tf.slice(labels, [0, 0, pitch_size], [-1, -1, bar_size])
        bar_logits = tf.slice(logits, [0, 0, pitch_size], [-1, -1, bar_size])
        self._bar_logits = bar_logits = tf.nn.softmax(bar_logits, "bar_logits")
        bar_logits_flat = tf.reshape(bar_logits, [-1, bar_size])
        self._bar_loss = bar_loss = tf.reduce_mean(-bar_labels_flat * tf.log(bar_logits_flat), name="bar_loss")
        tf.summary.scalar('bar_loss', bar_loss)

        self._loss = loss = config.pitch_loss_wight * pitch_loss + bar_loss
        tf.summary.scalar('loss', loss)

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients,
                                                      config.clip_norm)
        for param, gradient in zip(params, clipped_gradients):
            abs_gradient = tf.abs(gradient)
            # not valid contain ":"
            # variable name is "....:0"
            # TODO: Every name support. This is not fulfill such as "....:10".
            with tf.name_scope(param.name[:-2]):
                tf.summary.scalar('gradient/max', tf.reduce_max(abs_gradient))
                tf.summary.scalar('gradient/min', tf.reduce_min(abs_gradient))
                tf.summary.scalar('gradient/mean', tf.reduce_mean(abs_gradient))
        self._train_optimizer = config.optimizer_function.apply_gradients(zip(clipped_gradients, params),
                                                                          global_step)

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
