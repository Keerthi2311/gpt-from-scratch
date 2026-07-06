"""
train_demo.py — proof the engine works end to end.

Trains a tiny MLP on 4 hand-made examples (Karpathy's classic demo from the
"spelled-out intro to neural networks" video). Run:

    python train_demo.py

You should see the loss shrink toward ~0 and predictions approach the targets.
"""

from micrograd.engine import Value
from micrograd.nn import MLP
import random

random.seed(42)  # reproducible weights

# --- data: 4 examples, 3 features each, with target outputs ---
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # desired outputs

# --- model: 3 inputs -> 4 -> 4 -> 1 output (41 parameters total) ---
model = MLP(3, [4, 4, 1])
print(model)
print(f"number of parameters: {len(model.parameters())}\n")

LEARNING_RATE = 0.05

for step in range(100):

    # 1) FORWARD: run every example through the net, compute total
    #    squared-error loss. Every operation here is building the graph.
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # 2) ZERO GRADS: gradients accumulate (the += in engine.py),
    #    so wipe last step's gradients before computing new ones.
    model.zero_grad()

    # 3) BACKWARD: fill in .grad for all 41 parameters via the chain rule.
    loss.backward()

    # 4) UPDATE: nudge each parameter AGAINST its gradient.
    #    The gradient points uphill (direction of increasing loss),
    #    so we subtract to go downhill. This is gradient descent.
    for p in model.parameters():
        p.data -= LEARNING_RATE * p.grad

    if step % 10 == 0 or step == 99:
        print(f"step {step:3d} | loss {loss.data:.6f}")

print("\ntargets:    ", ys)
print("predictions:", [round(y.data, 3) for y in ypred])


x = Value(1.0)
y = (3*x).tanh()
y.backward()
print("prediction  y.data =", y.data)   # 0.995...  ← what tanh outputs
print("gradient    x.grad =", x.grad)   # 0.0296... ← how sensitive to x