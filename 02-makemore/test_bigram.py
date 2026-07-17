"""
test_bigram.py — checks on the counting bigram model.

The model is small enough that its numbers can be verified by hand. These
tests do exactly that: pin the vocabulary, confirm every row of P is a real
probability distribution, check the NLL against a from-scratch computation on
a tiny corpus, prove smoothing removes the -inf that unseen pairs would cause,
and confirm sampling is reproducible under a seed.

Pure Python, no dependencies.  Run:  python test_bigram.py
"""

import math
import random

from bigram import BigramCounts, build_vocab

results = []


def check(name, ok):
    print(f"{name:38s} {'PASS' if ok else 'FAIL'}")
    results.append(ok)


# --- fixtures ---------------------------------------------------------------
TINY = ["emma", "ava"]
model = BigramCounts(TINY, smoothing=1.0)

# 1) vocab: '.' plus the unique letters of the corpus, '.' pinned at 0.
stoi, itos = build_vocab(TINY)
check("vocab: '.' at index 0", itos[0] == ".")
check("vocab: sorted unique chars", itos == [".", "a", "e", "m", "v"])

# 2) every row of P is a probability distribution (sums to 1, all positive).
rows_ok = all(abs(sum(row) - 1.0) < 1e-12 for row in model.P)
positive = all(p > 0 for row in model.P for p in row)
check("P rows sum to 1", rows_ok)
check("P entries strictly positive", positive)

# 3) NLL matches a hand computation. For "emma": (. e)(e m)(m m)(m a)(a .);
#    for "ava": (. a)(a v)(v a)(a .). Recompute from P and compare.
def manual_nll(words, m):
    ll, n = 0.0, 0
    for w in words:
        chs = ["."] + list(w) + ["."]
        for a, b in zip(chs, chs[1:]):
            ll += math.log(m.P[m.stoi[a]][m.stoi[b]])
            n += 1
    return -ll / n

check("NLL matches manual sum", abs(model.nll(TINY) - manual_nll(TINY, model)) < 1e-12)

# 4) smoothing is what keeps NLL finite. With smoothing=0 an unseen pair has
#    probability 0 and log(0) = -inf; with smoothing it stays finite.
raw = BigramCounts(TINY, smoothing=0.0)
# ('v','v') never occurs in TINY, so its raw probability is exactly 0.
vv = raw.P[raw.stoi["v"]][raw.stoi["v"]]
check("unsmoothed: unseen pair has p=0", vv == 0.0)
check("smoothed: no zero probabilities", all(p > 0 for row in model.P for p in row))

# 5) sampling is deterministic given a seeded RNG (reproducible experiments).
a = [model.sample(random.Random(7)) for _ in range(3)]
b = [model.sample(random.Random(7)) for _ in range(3)]
check("sampling reproducible under seed", a == b)

# 6) counts are exact: 'm'->'m' happens once (the double-m in "emma").
check("count of (m -> m) is 1", model.N[model.stoi["m"]][model.stoi["m"]] == 1)
# 'a' ends a name in both "emma" and "ava" -> (a -> .) happens twice.
check("count of (a -> .) is 2", model.N[model.stoi["a"]][model.stoi["."]] == 2)

print()
if all(results):
    print(f"ALL {len(results)} CHECKS PASSED")
else:
    raise SystemExit("SOME CHECKS FAILED")
