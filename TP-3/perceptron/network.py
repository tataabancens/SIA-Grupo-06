def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def train(network, error_f, error_f_prime, inputs, expected_output, epochs=1000, learning_rate=0.01, verbose=True):
    for e in range(epochs):
        error = 0
        for x, y in zip(inputs, expected_output):
            # forward
            output = predict(network, x)

            # error
            error += error_f(y, output)

            # backward
            grad = error_f_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(inputs)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")
