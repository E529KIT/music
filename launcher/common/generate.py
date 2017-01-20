
def generate(session, model, start_inputs, size):
    state = session.run(model.initial_state)
    fetches = [model.logits, model.last_state]
    result = [start_inputs[0][0]]
    for input_ in start_inputs:
        feed_dict = {model.inputs: [input_], model.initial_state: state}
        logits, state = session.run(fetches, feed_dict)
        result.append(logits[0])

    for i in range(size - len(start_inputs)):
        feed_dict = {model.inputs: [[result[-1]]], model.initial_state: state}
        logits, state = session.run(fetches, feed_dict)
        result.append(logits[0])
    return result