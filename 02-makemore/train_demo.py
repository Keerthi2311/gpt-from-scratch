"""
train_demo.py — the counting bigram model, end to end.

Loads the 32k names, builds the bigram counts, reports the model's quality
(average NLL), and samples fresh names from it. No training loop, no
dependencies — the "training" for a counting model is just tallying pairs.

Run:  python train_demo.py
"""

import random

from bigram import BigramCounts, load_words

words = load_words()
model = BigramCounts(words, smoothing=1.0)

print(f"loaded {len(words)} names, vocab size {len(model.itos)}")
print(f"shortest {min(len(w) for w in words)}, longest {max(len(w) for w in words)}\n")

print("most common bigrams:")
for c, a, b in model.top_bigrams(8):
    print(f"  {a}{b}  {c}")

print(f"\naverage NLL over the training set: {model.nll(words):.4f}")
print("(uniform random would score log(27) = 3.2958; lower is better)\n")

# Sample names. They look bad — this is the point. A bigram only ever looks
# one character back, so it produces pronounceable-ish fragments but no memory
# of anything earlier in the word. Later stages fix exactly this.
rng = random.Random(2147483647)
print("20 names sampled from the model:")
for _ in range(20):
    print("  ", model.sample(rng))
