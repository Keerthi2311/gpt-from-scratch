"""
nn.py — a tiny neural network library built on top of the Value engine.

Because Value overloads +, *, etc., we can write a neural net as ordinary
Python arithmetic and get gradients for free. This mirrors PyTorch's design:
Module -> Neuron -> Layer -> MLP, each exposing .parameters().

Original code by Andrej Karpathy (MIT license). Comments added for study.
"""

import random
from micrograd.engine import Value


class Module:
    """Base class, like torch.nn.Module. Gives every model two abilities."""

    def zero_grad(self):
        # Gradients ACCUMULATE across backward() calls (remember the += in
        # engine.py). So before each new backward pass we must reset them,
        # or step 2's gradients would pile on top of step 1's.
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        # Every trainable Value in the model. Subclasses override this.
        return []


class Neuron(Module):
    """One neuron: out = relu(w · x + b)."""

    def __init__(self, nin, nonlin=True):
        # nin = number of inputs. One weight per input, initialized to
        # small random values so different neurons learn different things
        # (if all weights started identical, they'd stay identical — the
        # "symmetry breaking" problem).
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)  # bias: lets the neuron shift its threshold
        self.nonlin = nonlin

    def __call__(self, x):
        # Dot product w·x plus bias. sum(..., self.b) uses b as the
        # starting value of the sum — a neat trick to fold in the bias.
        # Every one of these * and + calls is building the autograd graph.
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        # Apply ReLU unless this is meant to be a linear (output) neuron.
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    """A layer = several neurons that all see the same input vector."""

    def __init__(self, nin, nout, **kwargs):
        # nout independent neurons, each taking nin inputs.
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        # Convenience: a 1-neuron layer returns a scalar, not a 1-item list.
        return out[0] if len(out) == 1 else out

    def parameters(self):
        # Flatten all neurons' parameters into one list.
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """Multi-Layer Perceptron: layers chained one after another.

    MLP(3, [4, 4, 1]) means: 3 inputs -> layer of 4 -> layer of 4 -> 1 output.
    """

    def __init__(self, nin, nouts):
        sz = [nin] + nouts  # e.g. [3, 4, 4, 1] — sizes of every "stage"
        # Consecutive pairs define each layer's (in, out) shape.
        # The LAST layer is linear (nonlin=False): for regression or for
        # feeding raw scores into a loss, we don't want ReLU clipping
        # negative outputs to zero.
        self.layers = [
            Layer(sz[i], sz[i+1], nonlin=(i != len(nouts)-1))
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        # Forward pass: output of each layer becomes input to the next.
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
