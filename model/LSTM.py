import tensorflow as tf


class DefaultConfig:
    batch_size = 100
    sequence_length = 128
    input_size = 20
    label_size = 20
    cell_size_list = [20, 30, 40]
    keep_prob = 0.9
    optimizer_function = tf.train.AdamOptimizer(0.5)
    clip_norm = 3


class LSTM:
    def __init__(self, is_train, config=DefaultConfig, generate=False, last_activation_function=tf.nn.softmax):
        self.batch_size = batch_size = config.batch_size
        self.sequence_length = sequence_length = config.sequence_length
        input_size = config.input_size
        label_size = config.label_size

        self._global_step = global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)

        if generate:
            self._inputs = inputs = tf.placeholder(tf.float32, [batch_size, None, input_size],
                                                   "generate_input")
        else:
            self._inputs = inputs = tf.placeholder(tf.float32, [None, sequence_length, input_size],
                                                   "input_data")
        self._labels = labels = tf.placeholder(tf.float32, [None, sequence_length, label_size], "labels")

        cells = []
        for cell_size in config.cell_size_list:
            cell = tf.nn.rnn_cell.BasicLSTMCell(cell_size)
            if is_train and config.keep_prob < 1:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
            cells.append(cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        self._initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, [sequence_length] * batch_size, self._initial_state,
                                           parallel_iterations=1, swap_memory=True)
        self._last_state = state

        outputs_size = config.cell_size_list[-1]
        outputs_flat = tf.reshape(outputs, [-1, outputs_size])
        self._logits = logits = tf.contrib.layers.fully_connected(inputs=outputs_flat, num_outputs=label_size,
                                                                  activation_fn=last_activation_function)
        self._logits_flat = logits_flat = tf.reshape(logits, [-1, label_size])

        self._labels_flat = labels_flat = tf.reshape(labels, [-1, label_size])
        self._loss = loss = tf.reduce_mean(tf.square(logits_flat - labels_flat))
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients,
                                                      config.clip_norm)
        abs_gradients = tf.reduce_mean(tf.abs(gradients))
        abs_clipped_gradients = tf.reduce_mean(tf.abs(clipped_gradients))
        tf.summary.scalar("gradients", abs_gradients)
        tf.summary.scalar("clipped_gradients", abs_clipped_gradients)
        tf.summary.scalar('loss', loss)
        self._train_optimizer = config.optimizer_function.apply_gradients(zip(clipped_gradients, params),
                                                                          global_step)

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def labels_flat(self):
        return self._labels_flat

    @property
    def initial_state(self):
        return self._initial_state

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
    def logits(self):
        return self._logits

    @property
    def logits_flat(self):
        return self._logits_flat
