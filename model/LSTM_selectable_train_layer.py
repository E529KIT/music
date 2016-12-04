import tensorflow as tf

class DefaultConfig:
    batch_size = 100
    sequence_length = 128
    input_size = 20
    label_size = 20
    cell_size_list = [(20, False), (30, False), (40, True)]
    keep_prob = 0.9
    optimizer_function = tf.train.AdamOptimizer(0.5)


class LSTM:
    def __init__(self, is_train, config=DefaultConfig):
        self.batch_size = batch_size = config.batch_size
        self.sequence_length = sequence_length = config.sequence_length
        input_size = config.input_size
        label_size = config.label_size

        self._global_step = global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)

        self._inputs = inputs = tf.placeholder(tf.float32, [None, sequence_length, input_size],
                                               "input_data")
        self._labels = labels = tf.placeholder(tf.float32, [None, sequence_length, label_size], "labels")

        train_variables = []
        states = []
        outputs = inputs
        initial_state_list = []
        for i, cell_info in enumerate(config.cell_size_list):
            cell_size, trainable = cell_info
            with tf.variable_scope('LSTM_cell%d' % i) as vs:
                cell = tf.nn.rnn_cell.BasicLSTMCell(cell_size, forget_bias=0.0)
                if is_train:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
                initial_state = cell.zero_state(batch_size, tf.float32)
                initial_state_list.append(initial_state)
                outputs, state = _connect_sequence(cell, outputs, initial_state,
                                                   sequence_length)
                states.append(state)
                if trainable:
                    train_variables += tf.get_collection(tf.GraphKeys.VARIABLES, scope=vs.name)

        self._last_state = tuple(states)
        self._initial_state = tuple(initial_state_list)
        outputs_size, _ = config.cell_size_list[-1]
        outputs_flat = tf.reshape(outputs, [-1, outputs_size])
        softmax_w = tf.get_variable(
            "weigth", [outputs_size, label_size], dtype=tf.float32)
        train_variables.append(softmax_w)
        softmax_b = tf.get_variable("bias", [label_size], dtype=tf.float32)
        train_variables.append(softmax_b)
        self._logits = logits = tf.matmul(outputs_flat, softmax_w) + softmax_b

        self._logits_flat = logits_flat = tf.reshape(logits, [-1, label_size])

        self._labels_flat = labels_flat = tf.reshape(labels, [-1, label_size])
        self._loss = loss = tf.reduce_mean(tf.square(logits_flat - labels_flat))
        tf.scalar_summary('loss', loss)
        gradients = tf.gradients(loss, train_variables)
        self._train_optimizer = config.optimizer_function.apply_gradients(zip(gradients, train_variables), global_step)

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


def _connect_sequence(cell, inputs, state, sequence_length):
    outputs = []
    for time_step in range(sequence_length):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        output, state = cell(inputs[:, time_step, :], state)
        outputs.append(output)
    return tf.pack(outputs), state
