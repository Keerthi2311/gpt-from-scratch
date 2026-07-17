# 02 — makemore: the bigram language model

Stage 2 of building GPT from first principles, following [Andrej Karpathy's
Zero to Hero series](https://karpathy.ai/zero-to-hero.html) — the first half of
[makemore](https://github.com/karpathy/makemore). The dataset is `names.txt`
(32,033 names, one per line); the goal is a model that *makes more* names like
them. This stage builds the simplest possible one — a bigram model — two
different ways, and shows they arrive at the same answer.

## What's here

```
bigram.py          # BigramCounts: learn P(next|current) by counting. Pure Python.
neural_bigram.py   # the same model learned by gradient descent (needs numpy)
train_demo.py       # counts, scores, and samples names from the counting model
test_bigram.py      # 10 checks: distributions, NLL, smoothing, sampling
names.txt           # 32,033 names
build.ipynb         # the exploratory notebook this stage grew out of
```

Run from this folder:

```bash
python train_demo.py     # the counting model, end to end
python test_bigram.py     # the checks
python neural_bigram.py   # the gradient-descent version
```

The counting model has **no dependencies**. `neural_bigram.py` uses numpy —
the only dependency in this stage.

## The bigram model in one sentence

Given the current character, predict the next one — looking back exactly one
character, nothing more. A name is bracketed with a boundary token `.` (index
0, marking both start and end), so `emma` is the bigrams `(. e) (e m) (m m)
(m a) (a .)`. Learn `P(next | current)` and you can both *score* a name (how
likely was it?) and *sample* a new one (walk the chain from `.` until you land
back on `.`).

## Two ways to the same P

**By counting** (`bigram.py`). Tally every pair into a 27×27 matrix, add-k
smooth, normalize each row. No training — the probabilities are read straight
off the counts. Average NLL on the training set: **2.4546**.

**By gradient descent** (`neural_bigram.py`). A single 27×27 weight matrix `W`;
`softmax(onehot(x) @ W)` is the predicted distribution; minimize mean negative
log likelihood. After 200 steps it converges to **~2.48** — the same model,
reached by optimization instead of arithmetic. That equivalence is the whole
point of makemore part 1: counting *is* the closed-form solution the gradient
is climbing toward.

The gradient it climbs is the one derived by hand back in
[stage 01](../01-micrograd/): for softmax followed by NLL,

```
d(loss)/d(logit_j) = p_j - 1{j is the true next char}
```

That single line is `dlogits` in `neural_bigram.py` — the predicted
probabilities with 1 subtracted off the true class. Stage 01 built `exp` and
`log` and verified this on two classes; stage 02 is where it does real work
across all 27.

## Three ideas this stage makes concrete

1. **`log(0)` is why smoothing exists.** An unseen pair (say `j` → `q`) has
   count 0, probability 0, and `log(0) = -inf` — one held-out example with
   that pair sends the whole NLL to infinity. Add-k smoothing keeps every
   probability strictly positive. In the neural version the same job is done
   by L2 regularization on `W`: both nudge the distribution toward uniform.

2. **NLL is the number that matters, not the samples.** The sampled names look
   terrible (`erauirabenerey`, `mai`, `depashlma`) and that is *correct* — a
   bigram has no memory beyond the previous character. The loss, 2.45 vs.
   log(27) = 3.30 for uniform, is what says the model actually learned
   something. Judge models by held-out loss, not vibes.

3. **A neural net can just re-derive a lookup table.** With one linear layer
   and one-hot inputs, `onehot(x) @ W` selects row `x` of `W`, and softmax of
   that row is the predicted distribution. There is nothing a bigram net can
   learn that the counts don't already contain — which is exactly why the two
   losses match, and exactly the ceiling the next stages break by looking
   further back than one character.

## Credits

Andrej Karpathy — [makemore](https://github.com/karpathy/makemore) and
[The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo).
