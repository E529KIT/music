# coding=utf-8
import datetime
import os
import random

import numpy as np
import tensorflow as tf


def _init(session, saver, logdir, load_dir=None):
    if load_dir:
        saver.restore(session, load_dir + "/data/model")

    save_file = logdir + "/data/model"
    if not os.path.isdir(logdir + "/data"):
        os.mkdir(logdir + "/data")
        session.run(tf.global_variables_initializer())
    elif load_dir is None:
        saver.restore(session, save_file)


def train(session, model, inputs, labels, epoch_num, saver, logdir, load_dir=None):
    train_writer = tf.summary.FileWriter(logdir, session.graph)
    merged = tf.summary.merge_all()
    _init(session, logdir, load_dir)

    save_file = logdir + "/data/model"
    epoch = 0
    while (True):
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


def train_has_batch(session, model, train_data, epoch_num, batch_size, saver, logdir, load_dir=None):
    '''
    inputs and labels shape = [data_size, one_data_size, sequence_length, input_size]
    one_data_sizeはdataごとに異なっていても良い
    inputsとlabelsからランダムにbatch_size個取り出して、epoch回学習する。
    '''
    train_writer = tf.summary.FileWriter(logdir, session.graph)
    merged = tf.summary.merge_all()
    _init(session, saver, logdir, load_dir)

    save_file = logdir + "/data/model"
    epoch = 0
    feed_data = random.sample(train_data, batch_size)
    feed_length = np.array(map(lambda x: len(x[0]), feed_data))
    feed_index = np.zeros(batch_size, int)

    global_step, state = session.run([model.global_step, model.initial_state])
    while True:
        inputs = [a[0][i] for a, i in zip(feed_data, feed_index)]
        labels = [a[1][i] for a, i in zip(feed_data, feed_index)]
        feed_dict = {model.inputs: inputs, model.labels: labels}
        for i, (c, h) in enumerate(model.initial_state):
            # 新しいdataになった場合、前のデータを使用しない（初期state(zero_state)を使う）
            if not feed_index[i] == 0:
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h

        if global_step % 100 == 0:
            fetches = {
                "train_opt": model.train_optimizer,
                "global_step": model.global_step,
                "last_state": model.last_state,
                "loss": model.loss,
                "summaries": merged
            }
            vals = session.run(fetches, feed_dict)
            global_step = vals["global_step"]
            state = vals["last_state"]

            train_writer.add_summary(vals["summaries"], global_step)
            now = datetime.datetime.today()
            print 'epoch:%d, step:%d, loss:%f, time:%s' % (epoch, global_step, vals["loss"], now.isoformat())
            saver.save(session, save_file)
        else:
            _, global_step, state = session.run([model.train_optimizer, model.global_step, model.last_state],
                                                feed_dict)
        feed_index += 1

        # update feed data
        change_feed_index = [i for i, change in enumerate(feed_length <= feed_index) if change]
        for index in change_feed_index:
            if len(train_data) == batch_size:
                feed_length[index] = 0
            else:
                add_data = random.choice(train_data)
                feed_data[index] = add_data
                feed_length[index] = len(add_data)
                feed_index[index] = 0

        if epoch_num > 0 and epoch_num <= epoch:
            break

    saver.save(session, save_file)
