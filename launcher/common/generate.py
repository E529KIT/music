def generate(session, model, start_input, size):
    state = session.run(model.initial_state)
    fetches = [model.logits, model.last_state]
    result = []
    logits = [start_input]
    for i in range(size):
        feed_dict = {model.inputs: [logits], model.initial_state: state}
        logits, state = session.run(fetches, feed_dict)
        result.append(logits)
    return result
