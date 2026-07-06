# 01 — micrograd: backprop from scratch

Stage 1 of building GPT from first principles, following [Andrej Karpathy's
Zero to Hero series](https://karpathy.ai/zero-to-hero.html). The core engine
(`engine.py`, `nn.py`) is Karpathy's [micrograd](https://github.com/karpathy/micrograd)
(MIT license), studied line by line and annotated. The extensions, tests,
training demo, and notes are mine.

## What's here

```
micrograd/
  engine.py         # Value: a scalar autograd engine (+ my extensions)
  nn.py             # Neuron -> Layer -> MLP on top of Value
train_demo.py       # trains a 41-parameter MLP; loss 4.74 -> 0.0002
test_expressions.py # 8 gradient checks on composed expressions
```

No dependencies — plain Python. Run from this folder:

```bash
python train_demo.py
python test_expressions.py
```

## What I added beyond the original

**Four ops in `engine.py`**: `exp`, `log`, `tanh`, `sigmoid` — each with its
backward closure. Derived the local derivatives, debugged them (my first
attempt put the gradient code after `return`, which taught me why gradients
live in closures fired later by `backward()`, not computed inline), and
verified every one against centered numerical differentiation.

**Expression tests** (`test_expressions.py`) that go beyond single-op checks:

- a value used three times in one expression — catches `=` vs `+=` gradient
  accumulation bugs that single-op tests miss
- a dead ReLU (gradient exactly 0 on the negative side)
- a saturated sigmoid (gradient ~0.0005 at x=10) — the vanishing gradient
  problem, demonstrated in one line
- `exp().log()` cancellation as a self-checking test
- softmax + negative log likelihood for two classes, verified against the
  theoretical gradient `p - 1` — the loss function every language model
  trains on, built from my own `exp` and `log`

**One deliberate failure**: flipping the update to `p.data += lr * p.grad`
walks uphill; steeper loss means bigger gradients means bigger steps, and the
loss overflows to `inf` then `NaN` within ~40 steps. Same feedback loop as a
too-large learning rate — now I know what a NaN'd training run means.

## The training loop (the five beats)

```
forward -> loss -> zero_grad -> backward -> update
```

Forward builds the graph; loss compares prediction to ground truth;
zero_grad wipes accumulated gradients (skip it and results look fine while
being silently wrong); backward runs the chain rule through the graph in
reverse topological order; update steps every parameter downhill:
`p.data -= lr * p.grad`.

## Three ideas I had to actually earn

1. **Gradients accumulate.** A node used in multiple places receives
   gradient along every path; `+=` sums them, `=` silently drops all but
   one. This is also why `zero_grad()` must run every iteration.
2. **`_backward` is a recipe, not a computation.** Each op attaches a
   closure capturing its local derivative; nothing fires until
   `backward()` walks the graph output-to-input in topological order.
3. **`.data` and `.grad` are different questions.** Forward answers "what
   does this compute?"; backward answers "how sensitive is the output to
   this value?" — tanh(1) ≈ 0.76 but its gradient is 1 − 0.76² ≈ 0.42.

## Credits

Andrej Karpathy — [micrograd](https://github.com/karpathy/micrograd) and the
[spelled-out intro to neural networks](https://www.youtube.com/watch?v=VMj-3S1tku0).

