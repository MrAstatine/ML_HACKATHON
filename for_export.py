# %%
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm
import string
import pickle   # for saving trained HMMs
import os


# %%
# === Load and preprocess corpus ===

def load_corpus(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().splitlines()

    # Clean words: lowercase, alphabetic only
    words = []
    for w in raw:
        w = w.strip().lower()
        # remove non-alpha chars
        w = ''.join([ch for ch in w if ch.isalpha()])
        if len(w) > 0:
            words.append(w)
    
    return words

# Load your corpus
corpus_path = "./Data/corpus.txt"
words = load_corpus(corpus_path)

print(f"✅ Loaded {len(words):,} words from corpus.")
print("Sample words:", words[:10])

# %%
# Compute word length distribution
length_counts = Counter(len(w) for w in words)

df_lengths = pd.DataFrame(list(length_counts.items()), columns=["Word Length", "Count"]).sort_values("Word Length")
display(df_lengths)

# Optional: visualize if you want to see spread
try:
    import matplotlib.pyplot as plt
    plt.bar(df_lengths["Word Length"], df_lengths["Count"])
    plt.xlabel("Word Length")
    plt.ylabel("Count")
    plt.title("Word Length Distribution in Corpus")
    plt.show()
except ImportError:
    print("matplotlib not installed — skipping visualization.")


# %%
# === Step 4A: Prepare training sequences for each word length ===

def prepare_sequences_by_length(words):
    seqs_by_len = defaultdict(list)
    for w in words:
        L = len(w)
        seq = ['^'] + list(w) + ['$']  # add start & end tokens
        seqs_by_len[L].append(seq)
    return seqs_by_len

seqs_by_len = prepare_sequences_by_length(words)
print(f"Prepared sequences for {len(seqs_by_len)} different lengths.")
print(f"Example for length 5: {seqs_by_len[5][:5]}")


# %%
# === Step 4B (Updated): Train length-specific + bucket HMMs ===

def train_char_HMM(sequences, smoothing=2.0):
    """Train a simple character bigram HMM with Laplace smoothing."""
    symbols = ['^'] + list(string.ascii_lowercase) + ['$']
    idx = {s: i for i, s in enumerate(symbols)}
    n = len(symbols)

    A_counts = np.zeros((n, n))
    pi_counts = np.zeros(n)

    for seq in sequences:
        if len(seq) < 2:
            continue
        # first real letter contributes to pi (after ^)
        pi_counts[idx[seq[1]]] += 1
        for a, b in zip(seq, seq[1:]):
            A_counts[idx[a], idx[b]] += 1

    # Laplace (add-k) smoothing to avoid zero transitions
    A = (A_counts + smoothing) / (A_counts.sum(axis=1, keepdims=True) + smoothing * n)
    pi = (pi_counts + smoothing) / (pi_counts.sum() + smoothing * n)

    return {"A": A, "pi": pi, "symbols": symbols, "index": idx}


# ---- Configuration ----
MIN_COUNT = 500  # minimum examples to train a separate HMM
SHORT_BUCKET_MAX = 3
LONG_BUCKET_MIN = 18

# ---- Container for models ----
HMMs_by_length = {}
bucket_models = {}
print("Training HMMs...")

# --- 1. Short bucket ---
short_sequences = [s for L, seqs in seqs_by_len.items() if L <= SHORT_BUCKET_MAX for s in seqs]
if len(short_sequences) > 0:
    bucket_models["short"] = train_char_HMM(short_sequences, smoothing=2.0)
    print(f"✅ Trained SHORT bucket HMM (<= {SHORT_BUCKET_MAX}) with {len(short_sequences)} sequences.")

# --- 2. Long bucket ---
long_sequences = [s for L, seqs in seqs_by_len.items() if L >= LONG_BUCKET_MIN for s in seqs]
if len(long_sequences) > 0:
    bucket_models["long"] = train_char_HMM(long_sequences, smoothing=2.0)
    print(f"✅ Trained LONG bucket HMM (>= {LONG_BUCKET_MIN}) with {len(long_sequences)} sequences.")

# --- 3. Length-specific HMMs ---
for L, seqs in tqdm(seqs_by_len.items(), desc="Training per-length HMMs"):
    if SHORT_BUCKET_MAX < L < LONG_BUCKET_MIN and len(seqs) >= MIN_COUNT:
        HMMs_by_length[L] = train_char_HMM(seqs, smoothing=2.0)

print(f"✅ Trained {len(HMMs_by_length)} length-specific HMMs.")


# %%
# === Step 4C: Train global HMM ===
all_sequences = [s for seqs in seqs_by_len.values() for s in seqs]
HMM_global = train_char_HMM(all_sequences)
print("✅ Global HMM trained.")


# %%
import pickle, os
os.makedirs("models2", exist_ok=True)

save_bundle = {
    "by_length": HMMs_by_length,
    "buckets": bucket_models,
    "global": HMM_global
}

with open("models2/HMM_bundle.pkl", "wb") as f:
    pickle.dump(save_bundle, f)

print("✅ All HMMs (length-specific + bucket + global) saved successfully.")


# %%
def load_test_words(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().splitlines()
    words = []
    for w in raw:
        w = w.strip().lower()
        w = ''.join([ch for ch in w if ch.isalpha()])
        if len(w) > 0:
            words.append(w)
    return words

test_path = "./Data/test.txt"
test_words = load_test_words(test_path)
print(f"✅ Loaded {len(test_words):,} test words.")
print("Sample:", test_words[:10])


# %%
import numpy as np

def compute_log_likelihood(word, hmm):
    idx = hmm["index"]
    A = hmm["A"]
    seq = ['^'] + list(word) + ['$']
    logp = 0.0
    for a, b in zip(seq, seq[1:]):
        if a not in idx or b not in idx:
            return -np.inf  # unknown symbol
        logp += np.log(A[idx[a], idx[b]] + 1e-12)
    return logp


# %%
import pickle

# Load trained bundle if not already in memory
with open("models2/HMM_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

HMMs_by_length = bundle["by_length"]
bucket_models = bundle["buckets"]
HMM_global = bundle["global"]

def get_hmm_for_length(L):
    if L in HMMs_by_length:
        return HMMs_by_length[L]
    elif L <= 3 and "short" in bucket_models:
        return bucket_models["short"]
    elif L >= 18 and "long" in bucket_models:
        return bucket_models["long"]
    else:
        return HMM_global


def get_interpolated_hmm(L, k=1000):
    """Interpolate local (per-length or bucket) HMM with global HMM.
    alpha = N_L / (N_L + k) where N_L is number of sequences of length L.
    """
    N_L = len(seqs_by_len.get(L, []))
    alpha = float(N_L) / (N_L + k) if (N_L + k) > 0 else 0.0
    local_model = get_hmm_for_length(L)
    global_model = HMM_global

    # assume symbol/index align between models (they were trained over same alphabet)
    interp = {}
    interp["symbols"] = local_model["symbols"]
    interp["index"] = local_model["index"]
    interp["A"] = alpha * local_model["A"] + (1.0 - alpha) * global_model["A"]
    interp["pi"] = alpha * local_model["pi"] + (1.0 - alpha) * global_model["pi"]
    return interp


# %%
from tqdm import tqdm
import string
import numpy as np

letters = list(string.ascii_lowercase)
idx_letter = {l:i for i,l in enumerate(letters)}

# Confusion matrix for next-letter prediction
confusion = np.zeros((26, 26))
total_predictions = 0
correct_predictions = 0

log_likelihoods = []

# For classification report
true_labels = []
pred_labels = []

for w in tqdm(test_words, desc="Evaluating"):
    hmm = get_interpolated_hmm(len(w))
    ll = compute_log_likelihood(w, hmm)
    log_likelihoods.append(ll)

    # Per-position prediction
    seq = ['^'] + list(w) + ['$']
    A = hmm["A"]
    idx = hmm["index"]
    for a, b in zip(seq, seq[1:]):
        if a in idx and b in idx and a in string.ascii_lowercase and b in string.ascii_lowercase:
            # Predict next letter = argmax P(next | current)
            pred_idx = np.argmax(A[idx[a], [idx[l] for l in string.ascii_lowercase]])
            pred_letter = letters[pred_idx]
            true_letter = b
            confusion[idx_letter[true_letter], idx_letter[pred_letter]] += 1
            total_predictions += 1
            if pred_letter == true_letter:
                correct_predictions += 1
            # Gather for classification report
            true_labels.append(true_letter)
            pred_labels.append(pred_letter)

avg_log_likelihood = np.mean(log_likelihoods)
perplexity = np.exp(-avg_log_likelihood / np.mean([len(w)+2 for w in test_words]))
accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0


# %%
print("\n=== HMM Evaluation Report ===")
print(f"Total test words: {len(test_words):,}")
print(f"Average log-likelihood: {avg_log_likelihood:.4f}")
print(f"Perplexity (lower = better): {perplexity:.4f}")
print(f"Letter prediction accuracy: {accuracy*100:.2f}%")

# Detailed classification report
try:
    from sklearn.metrics import classification_report
    print("\n=== Per-Letter Classification Report ===")
    print(classification_report(true_labels, pred_labels, digits=3, zero_division=0))
except Exception as e:
    print("sklearn not available or error when computing classification report:", e)

# Example: show most likely next letters after 't' from global model
try:
    t_idx = HMM_global["index"]['t']
    next_probs = HMM_global["A"][t_idx]
    top_next = sorted(zip(HMM_global["symbols"], next_probs), key=lambda x: x[1], reverse=True)[:10]
    print("\nMost likely after 't':", top_next)
except Exception as e:
    print("Could not compute transition example:", e)


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Normalize confusion matrix by rows
conf_norm = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-9)
df_conf = pd.DataFrame(conf_norm, index=letters, columns=letters)

plt.figure(figsize=(12,10))
sns.heatmap(df_conf, cmap="Blues", xticklabels=letters, yticklabels=letters)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Letter Prediction Confusion Matrix (Normalized by True Letter)")
plt.show()


# %%
results_by_len = defaultdict(list)
for w, ll in zip(test_words, log_likelihoods):
    results_by_len[len(w)].append(ll)

print("\n=== Per-Length Log-Likelihood Summary ===")
for L, vals in sorted(results_by_len.items()):
    print(f"Length {L:2d}: {len(vals):5d} words | Avg logP = {np.mean(vals):.3f}")



