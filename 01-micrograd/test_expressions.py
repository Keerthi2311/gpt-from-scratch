"""
test_expressions.py — gradient checks on composed expressions.

Single-op tests can pass while graph-level bugs hide. These expressions
stress the parts that actually break: gradient accumulation across shared
nodes, deep chaining, and the ops added beyond the original engine
(exp, log, tanh, sigmoid). Each test compares backprop's answer against
a centered numerical derivative: (f(x+h) - f(x-h)) / 2h.

Run:  python test_expressions.py
"""

from micrograd.engine import Value

H = 1e-6
TOL = 1e-4


def check(name, expr_fn, x0):
    """Backprop gradient vs numerical gradient for a single-input expression."""
    x = Value(x0)
    y = expr_fn(x)
    y.backward()
    num = (expr_fn(Value(x0 + H)).data - expr_fn(Value(x0 - H)).data) / (2 * H)
    ok = abs(x.grad - num) < TOL
    print(f"{name:28s} backprop={x.grad:+.6f}  numerical={num:+.6f}  {'PASS' if ok else 'FAIL'}")
    return ok


results = []

# 1) tanh + chain rule. Inner is 0 at x=-0.5, tanh'(0)=1, times inner
#    derivative 2 -> gradient is exactly 2.0. No calculator needed.
results.append(check("tanh chain", lambda x: (2*x + 1).tanh(), -0.5))

# 2) relu's kink. Positive side: gradient 3. Negative side: gradient 0 —
#    a "dead ReLU": nothing flows back, the neuron learns nothing.
results.append(check("relu (active side)", lambda x: (x*3 - 2).relu(), 1.0))
results.append(check("relu (dead side)",   lambda x: (x*3 - 2).relu(), 0.5))

# 3) sigmoid saturation. sigmoid(10) ~ 0.99995, so s*(1-s) is nearly zero.
#    The vanishing gradient problem, demonstrated in one line.
results.append(check("sigmoid (saturated)", lambda x: (x*10).sigmoid(), 1.0))

# 4) shared node: x appears three times, so gradients from three paths
#    must ACCUMULATE. This is the test that catches '=' instead of '+='.
results.append(check("shared node (x used 3x)",
                     lambda x: x.tanh() + x.sigmoid() * x.exp(), 0.5))

# 5) self-checking: exp then log cancel, so the gradient must equal the
#    gradient of the inner expression alone.
results.append(check("exp/log cancel",
                     lambda x: ((x.relu() + x.tanh()) * x.sigmoid()).exp().log(), 0.8))
inner_x = Value(0.8)
inner_y = (inner_x.relu() + inner_x.tanh()) * inner_x.sigmoid()
inner_y.backward()
outer_x = Value(0.8)
outer_y = ((outer_x.relu() + outer_x.tanh()) * outer_x.sigmoid()).exp().log()
outer_y.backward()
cancel_ok = abs(inner_x.grad - outer_x.grad) < 1e-9
print(f"{'  inner vs outer gradient':28s} {inner_x.grad:+.6f} vs {outer_x.grad:+.6f}  "
      f"{'PASS' if cancel_ok else 'FAIL'}")
results.append(cancel_ok)

# 6) softmax + negative log likelihood, two classes. THE loss of language
#    models. Theory: d(loss)/d(logit_a) = p_a - 1, d(loss)/d(logit_b) = p_b.
a, b = Value(2.0), Value(1.0)
p = a.exp() / (a.exp() + b.exp())
loss = -p.log()
loss.backward()
p_a = p.data
p_b = 1 - p_a
nll_ok = abs(a.grad - (p_a - 1)) < TOL and abs(b.grad - p_b) < TOL
print(f"{'softmax + NLL':28s} da={a.grad:+.6f} (theory {p_a-1:+.6f})  "
      f"db={b.grad:+.6f} (theory {p_b:+.6f})  {'PASS' if nll_ok else 'FAIL'}")
results.append(nll_ok)

print()
if all(results):
    print(f"ALL {len(results)} CHECKS PASSED")
else:
    raise SystemExit("SOME CHECKS FAILED")