"""Backpropagation manual para aprender la compuerta XOR.

Arquitectura: 2 entradas -> 2 neuronas ocultas -> 1 salida.
La clase Value construye el grafo y propaga los gradientes de forma
automatica, siguiendo el enfoque de Micrograd usado en el Colab.
"""

import math
import random


class Value:
    def __init__(self, data, children=(), operation="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._operation = operation
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, (self, other), "+")

        def backward():
            self.grad += output.grad
            other.grad += output.grad

        output._backward = backward
        return output

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other), "*")

        def backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = backward
        return output

    def __rmul__(self, other):
        return self * other

    def __pow__(self, exponent):
        output = Value(self.data**exponent, (self,), f"**{exponent}")

        def backward():
            self.grad += exponent * self.data ** (exponent - 1) * output.grad

        output._backward = backward
        return output

    def __truediv__(self, other):
        return self * other**-1

    def sigmoid(self):
        value = self.data
        sigmoid_value = 1.0 / (1.0 + math.exp(-value))
        output = Value(sigmoid_value, (self,), "sigmoid")

        def backward():
            self.grad += sigmoid_value * (1.0 - sigmoid_value) * output.grad

        output._backward = backward
        return output

    def log(self):
        if self.data <= 0.0:
            raise ValueError("log requiere un argumento positivo")
        output = Value(math.log(self.data), (self,), "log")

        def backward():
            self.grad += output.grad / self.data

        output._backward = backward
        return output

    def backward(self):
        topology = []
        visited = set()

        def build_topology(value):
            if value not in visited:
                visited.add(value)
                for child in value._prev:
                    build_topology(child)
                topology.append(value)

        build_topology(self)
        self.grad = 1.0
        for value in reversed(topology):
            value._backward()


def binary_cross_entropy(prediction, target):
    """L = -(y log(y_pred) + (1-y) log(1-y_pred))."""
    target_value = target if isinstance(target, Value) else Value(target)
    return -(target_value * prediction.log() + (1.0 - target_value) * (1.0 - prediction).log())


class Neuron:
    def __init__(self, number_of_inputs, random_generator):
        self.weights = [
            Value(random_generator.uniform(-1.0, 1.0))
            for _ in range(number_of_inputs)
        ]
        self.bias = Value(random_generator.uniform(-1.0, 1.0))

    def __call__(self, inputs):
        weighted_sum = sum(
            (weight * value for weight, value in zip(self.weights, inputs)),
            self.bias,
        )
        return weighted_sum.sigmoid()

    def parameters(self):
        return self.weights + [self.bias]


class Layer:
    def __init__(self, number_of_inputs, number_of_outputs, random_generator):
        self.neurons = [
            Neuron(number_of_inputs, random_generator)
            for _ in range(number_of_outputs)
        ]

    def __call__(self, inputs):
        outputs = [neuron(inputs) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [parameter for neuron in self.neurons for parameter in neuron.parameters()]


class MLP:
    def __init__(self, number_of_inputs, layer_sizes, seed=7):
        random_generator = random.Random(seed)
        sizes = [number_of_inputs] + layer_sizes
        self.layers = [
            Layer(sizes[index], sizes[index + 1], random_generator)
            for index in range(len(layer_sizes))
        ]

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def parameters(self):
        return [parameter for layer in self.layers for parameter in layer.parameters()]


def main():
    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    targets = [0.0, 1.0, 1.0, 0.0]
    network = MLP(2, [2, 1], seed=0)
    parameters = network.parameters()
    learning_rate = 1.0
    epochs = 20_000

    for epoch in range(epochs):
        predictions = [network(row) for row in inputs]
        losses = [
            binary_cross_entropy(prediction, target)
            for prediction, target in zip(predictions, targets)
        ]
        loss = sum(losses) / len(losses)

        for parameter in parameters:
            parameter.grad = 0.0
        loss.backward()

        for parameter in parameters:
            parameter.data -= learning_rate * parameter.grad

        if epoch % 2_000 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:04d} | loss = {loss.data:.6f}")

    print("\nPredicciones finales:")
    for row, target in zip(inputs, targets):
        prediction = network(row)
        predicted_class = int(prediction.data >= 0.5)
        print(
            f"{int(row[0])} XOR {int(row[1])} -> "
            f"salida={prediction.data:.6f}, clase={predicted_class}, "
            f"objetivo={int(target)}"
        )

    print("\nCantidad de parametros:", len(parameters))


if __name__ == "__main__":
    main()