import json
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = np.random.uniform(-1, 1)

    def feedforward(self, inputs):
        total = np.dot(inputs, self.weights) + self.bias
        return sigmoid(total)

class LinearNeuron:
    def __init__(self, input_size):
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = np.random.uniform(-1, 1)

    def feedforward(self, inputs):
        return float(np.dot(inputs, self.weights) + self.bias)

class ClassificationNN:
    def __init__(self, input_size=4, hidden_size=8, output_size=3):
        self.hidden_layer = [Neuron(input_size) for _ in range(hidden_size)]
        self.output_layer = [Neuron(hidden_size) for _ in range(output_size)]

    def feedforward(self, inputs):
        h = np.array([n.feedforward(inputs) for n in self.hidden_layer])
        return np.array([n.feedforward(h) for n in self.output_layer])

    def predict(self, inputs):
        return int(np.argmax(self.feedforward(inputs)))

class RegressionNN:
    def __init__(self, input_size=3, hidden_size=8):
        self.hidden_layer = [Neuron(input_size) for _ in range(hidden_size)]
        self.output_neuron = LinearNeuron(hidden_size)

    def feedforward(self, inputs):
        h = np.array([n.feedforward(inputs) for n in self.hidden_layer])
        return self.output_neuron.feedforward(h)

def load_classification_model(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    net = ClassificationNN(4, 8, 3)
    for neuron, params in zip(net.hidden_layer, data["hidden"]):
        neuron.weights = np.array(params["weights"])
        neuron.bias = params["bias"]

    for neuron, params in zip(net.output_layer, data["output"]):
        neuron.weights = np.array(params["weights"])
        neuron.bias = params["bias"]

    mn = np.array(data["mn"])
    mx = np.array(data["mx"])
    class_names = data.get("class_names") or data.get("names")
    return net, mn, mx, class_names

def load_regression_model(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    net = RegressionNN(3, 8)
    for neuron, params in zip(net.hidden_layer, data["hidden"]):
        neuron.weights = np.array(params["weights"])
        neuron.bias = params["bias"]

    net.output_neuron.weights = np.array(data["output"]["weights"])
    net.output_neuron.bias = data["output"]["bias"]

    mn_x = np.array(data["mn_x"])
    mx_x = np.array(data["mx_x"])
    mn_y = float(data["mn_y"])
    mx_y = float(data["mx_y"])
    return net, mn_x, mx_x, mn_y, mx_y
