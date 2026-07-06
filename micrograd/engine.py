"""
engine.py — the heart of micrograd.

The whole idea: a `Value` wraps a single float, and every math operation
(+, *, **, relu) builds a graph recording HOW that number was computed.
Calling .backward() on the final output walks that graph in reverse and
fills in .grad for every Value using the chain rule.

That's all backpropagation is: bookkeeping + the chain rule.
Original code by Andrej Karpathy (MIT license). Comments added for study.
Extensions beyond the original (exp, log, tanh, sigmoid) added by Keerthi.
"""

import math


class Value:
    """ Stores a single scalar value and its gradient. """

    def __init__(self, data, _children=(), _op=''):
        self.data = data          # the actual number this node holds
        self.grad = 0             # d(final output)/d(this value), filled in by backward()

        # --- autograd graph bookkeeping ---
        # _backward: a function that knows how to push this node's gradient
        # back to its parents. Default is a no-op (leaf nodes have no parents).
        self._backward = lambda: None
        # _prev: the Values that were combined to produce this one.
        # e.g. for c = a + b, c._prev = {a, b}
        self._prev = set(_children)
        # _op: which operation made this node ('+', '*', 'ReLU'...) — only
        # used for visualization/debugging, not for the math.
        self._op = _op

    def __add__(self, other):
        # Allow `Value(2) + 3` by auto-wrapping plain numbers.
        other = other if isinstance(other, Value) else Value(other)
        # Forward pass: just add. But also record parents and the op.
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # Chain rule for addition: out = self + other
            # d(out)/d(self) = 1 and d(out)/d(other) = 1,
            # so each parent receives out.grad unchanged.
            #
            # Why += and not =? Because a Value can be used in MULTIPLE
            # places in the graph (e.g. b = a + a). Gradients from every
            # path must ACCUMULATE. Using = would silently overwrite one
            # path's contribution — the classic backprop bug.
            self.grad += out.grad
            other.grad += out.grad
        # Attach the recipe to the output node. It runs later, during backward().
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # Chain rule for multiplication: out = self * other
            # d(out)/d(self) = other.data  -> "the gradient of a product
            # w.r.t. one factor is the OTHER factor."
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        # Only constant powers (x**2, x**-1). Supporting Value**Value would
        # need the derivative of a^b w.r.t. b, which involves log — skipped
        # to keep the engine tiny.
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            # Power rule: d(x^n)/dx = n * x^(n-1), scaled by out.grad (chain rule).
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        # ReLU(x) = max(0, x). The nonlinearity — without it, stacked layers
        # would collapse into one big linear function.
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            # Derivative of ReLU: 1 where input was positive, 0 otherwise.
            # (out.data > 0) is a bool that Python treats as 1 or 0.
            # So the gradient either passes through untouched or is killed.
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    # ==================================================================
    # EXTENSIONS beyond Karpathy's original engine (added by Keerthi).
    # Same 5-line pattern as relu: forward compute -> record parents ->
    # attach a _backward closure holding the local derivative.
    # ==================================================================

    def exp(self):
        # e^x. Its derivative is itself — so the local derivative is
        # simply out.data, which we already computed in the forward pass.
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def log(self):
        # Natural log. Only defined for positive inputs — fail loudly now
        # rather than mysteriously mid-training.
        assert self.data > 0, "log requires positive input"
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            # d(ln x)/dx = 1/x
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        # tanh(x) = (e^2x - 1) / (e^2x + 1), squashes to (-1, 1).
        a = self.data
        t = (math.exp(2*a) - 1) / (math.exp(2*a) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d(tanh)/dx = 1 - tanh^2(x) = 1 - t^2 — expressible purely
            # in terms of the OUTPUT, so it's cheap: no re-computing exp.
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        # sigmoid(x) = 1 / (1 + e^-x), squashes to (0, 1).
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            # d(sigmoid)/dx = s * (1 - s) — again, output-only. This is
            # why sigmoid/tanh backward passes are so cheap in practice.
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        """Run backpropagation from this node (usually the loss)."""

        # Step 1: topological sort. We must process a node only AFTER
        # everything downstream of it has been processed, so that its
        # out.grad is complete when its _backward() runs. A DFS that
        # appends a node after visiting all its children gives us
        # exactly that order (leaves first, this node last).
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Step 2: seed the output. d(loss)/d(loss) = 1 — the base case
        # of the chain rule.
        self.grad = 1
        # Step 3: walk the graph BACKWARDS (output -> inputs), letting each
        # node distribute its gradient to its parents via its stored recipe.
        for v in reversed(topo):
            v._backward()

    # ------------------------------------------------------------------
    # Everything below is convenience: expressing -, /, and right-hand-side
    # operations in terms of the core ops (+, *, **) so their gradients
    # come for free. No new calculus needed.
    # ------------------------------------------------------------------

    def __neg__(self): # -self  ==  self * -1
        return self * -1

    def __radd__(self, other): # handles `3 + Value(2)` (int on the left)
        return self + other

    def __sub__(self, other): # a - b  ==  a + (-b)
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # handles `3 * Value(2)`
        return self * other

    def __truediv__(self, other): # a / b  ==  a * b^-1 (reuses the pow rule!)
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
