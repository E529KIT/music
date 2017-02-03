def generate(session, model, start_inputs, size, norm_func=None):
    state = session.run(model.initial_state)
    fetches = [model.logits, model.last_state]
    result = [start_inputs[0][0]]
    logits = []
    start_input_size = len(start_inputs)

    for input_ in start_inputs:
        feed_dict = {model.inputs: [input_], model.initial_state: state}
        logits, state = session.run(fetches, feed_dict)
        logits = logits[0][0]
        if norm_func is not None:
            logits = norm_func(logits)
        result.append(logits)

    for i in range(size - start_input_size):
        feed_dict = {model.inputs: [[logits]], model.initial_state: state}
        logits, state = session.run(fetches, feed_dict)
        logits = logits[0][0]
        if norm_func is not None:
            logits = norm_func(logits)
        result.append(logits)
    return result
