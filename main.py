from __future__ import annotations
from typing import Callable

from math import sqrt
import random
import time

import numpy


class Layer:
    def __init__(self, num_nodes: int, num_nodes_out: int, activation_func: Callable[[float], float]):
        self.num_nodes = num_nodes
        self.num_nodes_out = num_nodes_out
        self.weights = numpy.array(
            [[random.random() / sqrt(num_nodes) for __ in range(num_nodes_out)] for _ in range(num_nodes)]
        )
        self.biases = numpy.zeros(num_nodes)
        self.activation_func = activation_func
        # not actually lists, I don't know how to type-hint numpy arrays
        self.weights_cost_gradient: list[list[float]] | None = None
        self.biases_cost_gradient: list[float] | None = None

    def activate(self, inputs: list[float]):
        node_activations = [
            self.activation_func(inputs[node_i]) + self.biases[node_i]
            for node_i in range(self.num_nodes)
        ]
        return node_activations @ self.weights  # why is this not right-to-left like the math notation


class Network:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    @classmethod
    def build(cls, layer_sizes: tuple[int, ...]) -> Network:
        layers = [Layer(layer_sizes[i], layer_sizes[i + 1], lambda x: max(x, 0)) for i in range(len(layer_sizes) - 1)]
        return Network(layers)

    def get_output(self, inputs: list[float]) -> list[float]:
        for layer in self.layers:
            inputs = layer.activate(inputs)
        return inputs

    def get_cost(self, inputs: list[float], expected_outputs: list[float]) -> float:
        outputs = self.get_output(inputs)
        assert len(outputs) == len(expected_outputs)
        return sum((output - expected) ** 2 for output, expected in zip(outputs, expected_outputs))

    def get_average_cost(self, data_points: list[list[float]], expected_outputs: list[list[float]]) -> float:
        assert len(data_points) == len(expected_outputs)
        return sum(self.get_cost(inps, exps) for inps, exps in zip(data_points, expected_outputs)) / len(data_points)

    def learn(self, data_points: list[list[float]], expected_outputs: list[list[float]], learn_rate: float, n: int):
        # after putting in some more debugging prints, I realized the problem (or at least one of them) was
        # that I was doing the gradient descent after each layer's slope calculations instead of calculating the
        # whole gradient for the network first, THEN changing the weights and biases accordingly
        h = 0.0001
        start = time.time()
        for i in range(n):
            original_cost = self.get_average_cost(data_points, expected_outputs)
            for layer_i, layer in enumerate(self.layers, 1):
                layer.weights_cost_gradient = []
                layer.biases_cost_gradient = []
                for node_index in range(layer.num_nodes):
                    layer.biases[node_index] += h
                    cost_difference = self.get_average_cost(data_points, expected_outputs) - original_cost
                    layer.biases_cost_gradient.append(cost_difference / h)
                    weights_cost_slopes = []
                    for node_out_index in range(layer.num_nodes_out):
                        layer.weights[node_index][node_out_index] += h
                        cost_difference = self.get_average_cost(data_points, expected_outputs) - original_cost
                        weights_cost_slopes.append(cost_difference / h)
                        layer.weights[node_index][node_out_index] -= h
                    layer.weights_cost_gradient.append(weights_cost_slopes)

            # apply cost gradients to weights and biases
            for layer in self.layers:
                for node_index in range(layer.num_nodes):
                    layer.biases[node_index] -= layer.biases_cost_gradient[node_index] * learn_rate
                    for node_out_index in range(layer.num_nodes_out):
                        change = layer.weights_cost_gradient[node_index][node_out_index] * learn_rate
                        layer.weights[node_index][node_out_index] -= change
            if (i + 1) % 100 == 0:
                print(f"{self.get_output([1, 1])} cost {self.get_average_cost(data_points, expected_outputs)} "
                      f"({time.time() - start}s, {round((i + 1)/n * 100)}%)")
                start = time.time()


def main():
    network = Network.build((2, 2, 2))
    data_points = [
        [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0],
        [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1], [9, 1],
        [0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2], [9, 2],
        [0, 3], [1, 3], [2, 3], [3, 3], [4, 3], [5, 3], [6, 3], [7, 3], [8, 3], [9, 3],
        [0, 4], [1, 4], [2, 4], [3, 4], [4, 4], [5, 4], [6, 4], [7, 4], [8, 4], [9, 4],
        [0, 5], [1, 5], [2, 5], [3, 5], [4, 5], [5, 5], [6, 5], [7, 5], [8, 5], [9, 5],
        [0, 6], [1, 6], [2, 6], [3, 6], [4, 6], [5, 6], [6, 6], [7, 6], [8, 6], [9, 6],
    ]
    expected_outputs = [
        [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1],
        [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1],
        [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1],
        [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1],
        [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
        [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
        [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
    ]
    print(network.get_output([1, 1]), "cost", network.get_average_cost(data_points, expected_outputs))
    network.learn(data_points, expected_outputs, 0.01, 2000)
    for data_point, expected_output in zip(data_points, expected_outputs):
        print(f"{data_point} -> {network.get_output(data_point)} expected {expected_output}")

    print("\nnon-training examples:")
    for data_point, expected_output in (([0.5, 0.5], [1, 0]), ([3.346, 4.12], [1, 0]),
                                        ([7.99, 4.2], [0, 1]), ([8.55, 3.789], [0, 1])):
        print(f"{data_point} -> {network.get_output(data_point)} expected {expected_output}")


if __name__ == "__main__":
    main()
