import datetime
import os

import tensorflow as tf


def train(session, model, inputs, labels, epoch, saver, logdir):
    train_writer = tf.summary.FileWriter(logdir, session.graph)
    merged = tf.summary.merge_all()

    save_file = logdir + "/data/model"
    if not os.path.isdir(logdir + "/data"):
        os.mkdir(logdir + "/data")
        session.run(tf.global_variables_initializer())
    else:
        saver.restore(session, save_file)

    for i in range(epoch):
        global_step, state = session.run([model.global_step, model.initial_state])
        for input_, label in zip(inputs, labels):
            feed_dict = {model.inputs: [input_], model.initial_state: state, model.labels: [label]}
            if global_step % 100 == 0:
                fetches = [model.train_optimizer, model.global_step, model.last_state, model.loss, merged]
                _, global_step, state, loss, summaries = session.run(fetches, feed_dict)
                train_writer.add_summary(summaries, global_step)
                now = datetime.datetime.today()
                print 'epoch:%d, step:%d, loss:%f, time:%s' % (i, global_step, loss, now.isoformat())
                saver.save(session, save_file)
            else:
                _, global_step, state = session.run([model.train_optimizer, model.global_step, model.last_state],
                                                    feed_dict)
    saver.save(session, save_file)
