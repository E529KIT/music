import datetime
import os

import tensorflow as tf


def train(session, model, inputs, labels, epoch_num, saver, logdir, load_dir=None):
    if load_dir:
        saver.restore(session, load_dir + "/data/model")
    train_writer = tf.summary.FileWriter(logdir, session.graph)
    merged = tf.summary.merge_all()

    save_file = logdir + "/data/model"
    if not os.path.isdir(logdir + "/data"):
        os.mkdir(logdir + "/data")
        session.run(tf.global_variables_initializer())
    elif load_dir is None:
        saver.restore(session, save_file)

    epoch = 0
    while(True):
        global_step, state = session.run([model.global_step, model.initial_state])
        for input_, label in zip(inputs, labels):
            feed_dict = {model.inputs: [input_], model.initial_state: state, model.labels: [label]}
            if global_step % 100 == 0:
                fetches = [model.train_optimizer, model.global_step, model.last_state, model.loss, merged]
                _, global_step, state, loss, summaries = session.run(fetches, feed_dict)
                train_writer.add_summary(summaries, global_step)
                now = datetime.datetime.today()
                print 'epoch:%d, step:%d, loss:%f, time:%s' % (epoch, global_step, loss, now.isoformat())
                saver.save(session, save_file)
            else:
                _, global_step, state = session.run([model.train_optimizer, model.global_step, model.last_state],
                                                    feed_dict)
        epoch += 1
        if epoch_num > 0 and epoch_num <= epoch:
            break
    saver.save(session, save_file)
