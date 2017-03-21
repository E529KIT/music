import tensorflow as tf


def create_sequence_example(inputs, labels):
    input_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=input_))
        for input_ in inputs]
    label_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=label))
        for label in labels]
    feature_list = {
        'inputs': tf.train.FeatureList(feature=input_features),
        'labels': tf.train.FeatureList(feature=label_features)
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(feature_lists=feature_lists)


def get_padded_batch(file_list, batch_size, input_size, label_size, num_enqueuing_threads=4):
    file_queue = tf.train.string_input_producer(file_list)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)

    sequence_features = {
        'inputs': tf.FixedLenSequenceFeature(shape=[input_size],
                                             dtype=tf.float32),
        'labels': tf.FixedLenSequenceFeature(shape=[label_size],
                                             dtype=tf.float32)
    }

    context, sequence = tf.parse_single_sequence_example(
        serialized_example, sequence_features=sequence_features)

    length = tf.shape(sequence['inputs'])[0]

    queue = tf.PaddingFIFOQueue(
        capacity=1000,
        dtypes=[tf.float32, tf.float32, tf.int32],
        shapes=[(None, input_size), (None, label_size), ()])

    enqueue_ops = [queue.enqueue([sequence['inputs'],
                                  sequence['labels'],
                                  length])] * num_enqueuing_threads
    tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))
    return queue.dequeue_many(batch_size)
