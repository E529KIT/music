import datetime
import tensorflow as tf

def train(session, model, inputs, labels, epoch, saver, savepath):
    train_writer = tf.train.SummaryWriter('./log', session.graph)
    merged = tf.merge_all_summaries()
    session.run(tf.global_variables_initializer())
    global_step, state = session.run([model.global_step, model.initial_state])
    fetches = [model.train_optimizer, model.global_step, model.last_state, model.loss, merged]
    for i in range(epoch):
        for input_, label in zip(inputs, labels):
            feed_dict = {model.inputs: [input_], model.initial_state: state, model.labels: [label]}
            _, global_step, state, loss, summaries = session.run(fetches, feed_dict)
            train_writer.add_summary(summaries, global_step)
        if i % 50 == 0:
            saver.save(session, savepath)
            now = datetime.datetime.today()
            print '%d: %s' % (i, now.isoformat())


