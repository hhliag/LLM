import random

from engine import Value


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for i in range(nin)]
        self.b = Value(random.uniform(-1,1))
    def __call__(self, x):
        act = sum([xi*wi for xi, wi in zip(self.w, x)], self.b)
        out = act.tanh()
        return out
    def parameters(self):
        return self.w+[self.b]
class Layer:
    def __init__(self, nin, nneur):
        self.neurons = [Neuron(nin) for _ in range(nneur)]
    def __call__(self, x):
        outs = [neur(x) for neur in self.neurons]
        return outs[0] if len(outs)==1 else outs
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [l for layer in self.layers for l in layer.parameters()]
