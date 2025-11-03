# Hangman HMM + RL Notebook
# File: Hangman_HMM_RL_Notebook.py
# This file is written as a runnable script / notebook-style Python file.
# It implements:
# 1. A length-specific character n-gram model (used as the HMM-like oracle)
# 2. A Hangman environment suitable for RL training
# 3. A DQN agent (PyTorch) to learn guessing strategy using the oracle
# 4. Training and evaluation loops and plotting utilities

# Notes:
# - Place the provided corpus at Data/corpus.txt and the test words at Data/test.txt
# - The script defaults to using GPU if available for training the DQN; it will fall back to CPU.
# - The HMM portion is implemented as length-specific Markov models (1st and 2nd order n-grams)
#   because they capture positional/contextual letter probabilities and are simple, robust,
#   and fast to train on a 50k-word corpus. They serve as the agent's probabilistic oracle.

# --- Imports ---
import os
import random
import math
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# --- Utility functions ---

def load_wordlist(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        words = [w.strip().lower() for w in f if w.strip()]
    return words

# Enhanced letter frequency with strategic ordering
LETTER_FREQ = ['e', 'a', 'i', 'o', 't', 'n', 's', 'r', 'h', 'l', 'd', 'c', 'u', 'm', 'p', 'f', 'g', 'w', 'y', 'b', 'v', 'k', 'x', 'j', 'q', 'z']

def get_smart_first_guesses(masked_word: str, guessed: set) -> Optional[str]:
    """Strategic first guesses based on word patterns"""
    # If no guesses yet, start with most common vowel
    if len(guessed) == 0:
        return 'e'
    if len(guessed) == 1:
        return 'a'
    if len(guessed) == 2:
        return 'i'
    if len(guessed) == 3:
        return 'o'
    if len(guessed) == 4:
        return 't'
    if len(guessed) == 5:
        return 'n'
    return None

# --- HMM-like: Length-specific n-gram models ---
class LengthSpecificNGramOracle:
    """
    A simple length-specific character n-gram model used as a probabilistic oracle.
    Trains separate n-gram counts per word length (for lengths with at least min_examples).

    Methods:
    - fit(corpus): build counts
    - predict_mask_prob(masked_word, guessed_letters): returns a probability vector over alphabet
      representing probability of each letter being in any of the blank positions, normalized.
    """
    def __init__(self, n=2, min_examples=50):
        self.n = n  # order (1 = unigram, 2 = bigram, etc.)
        self.min_examples = min_examples
        self.alphabet = list("abcdefghijklmnopqrstuvwxyz")
        self.idx = {c:i for i,c in enumerate(self.alphabet)}
        # Data structure: for each length -> ngram counts
        self.length_models = {}  # length -> dict of contexts -> Counter of next char
        self.length_totals = {}  # length -> total counts per context

    def fit(self, words: List[str]):
        bylen = defaultdict(list)
        for w in words:
            bylen[len(w)].append(w)
        for L, ws in bylen.items():
            if len(ws) < self.min_examples:
                continue
            models = defaultdict(Counter)
            for w in ws:
                padded = ("^" * (self.n-1)) + w + "$"
                for i in range(self.n-1, len(padded)-1):
                    ctx = padded[i-(self.n-1):i]
                    nxt = padded[i]
                    models[ctx][nxt] += 1
            self.length_models[L] = models
            # compute totals for smoothing
            totals = {ctx: sum(cnts.values()) for ctx, cnts in models.items()}
            self.length_totals[L] = totals
        print(f"Trained length-specific n-gram oracle for lengths: {sorted(self.length_models.keys())}")

    def _prob_next_char(self, ctx: str, length: int) -> np.ndarray:
        """Return probability vector over alphabet for next char after context ctx for words of given length.
        We smooth with add-one (Laplace) smoothing and ignore boundary tokens when needed.
        """
        vocab = self.alphabet + ["$"]
        V = len(vocab)
        probs = np.ones(len(vocab))  # add-one smoothing
        if length in self.length_models and ctx in self.length_models[length]:
            counts = self.length_models[length][ctx]
            for i, ch in enumerate(vocab):
                probs[i] += counts.get(ch, 0)
            probs = probs / probs.sum()
        else:
            probs = probs / probs.sum()
        # return probabilities only for letters (exclude $)
        return probs[:-1]

    def predict_mask_prob(self, masked_word: str, guessed_letters: set) -> np.ndarray:
        """
        Given a masked word like '_pp_e', and set of guessed letters, return an
        alphabet-sized probability vector for each letter representing its
        likelihood of being in one of the blank positions (averaged across blanks).
        Enhanced with pattern rules and frequency blending.
        """
        L = len(masked_word)
        if L not in self.length_models:
            # backoff: use most similar available length (closest)
            if not self.length_models:
                return np.ones(26) / 26
            lengths = sorted(self.length_models.keys(), key=lambda x: abs(x-L))
            L = lengths[0]
        # For each blank position, estimate distribution conditioned on left context (and maybe right via reverse model)
        probs_accum = np.zeros(26)
        blanks = 0
        padded = ("^" * (self.n-1)) + masked_word + "$"
        for i in range(self.n-1, self.n-1+len(masked_word)):
            ch = padded[i]
            if ch == '_':
                blanks += 1
                ctx = padded[i-(self.n-1):i]
                # replace unknowns in ctx with most likely char (or keep as ^/_)
                # For simplicity, keep ctx as-is; our model trained on real chars and ^ padding
                # If ctx contains '_' we replace by '^' (backoff) to allow smoothing
                ctx_clean = ''.join([c if c != '_' else '^' for c in ctx])
                pnext = self._prob_next_char(ctx_clean, L)
                probs_accum += pnext
        if blanks == 0:
            # no blanks -> probabilities zero for all
            return np.zeros(26)
        probs = probs_accum / blanks
        
        # Apply pattern-based boosts BEFORE zeroing guessed letters
        # Q -> U pattern (very strong)
        if 'q' in masked_word and 'u' not in guessed_letters:
            probs[self.idx['u']] *= 20.0
        
        # Common ending patterns
        if masked_word.endswith('_') and len(masked_word) > 2:
            # Words often end in e, s, d, t, y
            for letter, boost in [('e', 3.0), ('s', 2.5), ('d', 2.0), ('t', 1.8), ('y', 1.5)]:
                if letter not in guessed_letters:
                    probs[self.idx[letter]] *= boost
        
        # _ng pattern -> likely 'i' before 'ng'
        if 'ng' in masked_word or '_n' in masked_word or 'n_' in masked_word:
            if 'i' not in guessed_letters:
                probs[self.idx['i']] *= 3.0
        
        # Double letter patterns - if we see a letter, boost itself
        for letter in set(masked_word) - {'_'}:
            if letter in self.idx and letter not in guessed_letters:
                # Same letter might appear again
                probs[self.idx[letter]] *= 1.5
        
        # _th pattern -> e (the, with, etc.)
        if 'th' in masked_word and 'e' not in guessed_letters:
            probs[self.idx['e']] *= 2.0
        
        # zero out already guessed letters to encourage new guesses
        for g in guessed_letters:
            if g in self.idx:
                probs[self.idx[g]] = 0.0
        # renormalize
        s = probs.sum()
        if s <= 0:
            return np.ones(26) / 26
        return probs / s

# --- Hangman Environment ---
class HangmanEnv:
    def __init__(self, word: str, max_wrong: int = 6):
        self.word = word.lower()
        self.max_wrong = max_wrong
        self.reset()

    def reset(self):
        self.guessed = set()
        self.wrong = 0
        self.mask = ['_' for _ in self.word]
        self.done = False
        return self._state()

    def _state(self):
        return ''.join(self.mask), set(self.guessed), self.max_wrong - self.wrong

    def step(self, letter: str) -> Tuple[Tuple[str, set, int], float, bool]:
        """
        Guess a letter. Returns (state, reward, done).
        Enhanced reward structure to strongly discourage mistakes.
        """
        letter = letter.lower()
        if self.done:
            raise RuntimeError('Game already finished')
        if letter in self.guessed:
            # repeated guess - very bad
            reward = -2.0
            done = False
            return self._state(), reward, self.done
        self.guessed.add(letter)
        if letter in self.word:
            # reveal positions
            num_revealed = 0
            for i, ch in enumerate(self.word):
                if ch == letter:
                    self.mask[i] = letter
                    num_revealed += 1
            if '_' not in self.mask:
                self.done = True
                # Big win bonus + bonus for remaining lives
                return self._state(), 50.0 + (self.max_wrong - self.wrong) * 5.0, True
            # Reward proportional to letters revealed
            return self._state(), 2.0 + num_revealed, False
        else:
            self.wrong += 1
            if self.wrong >= self.max_wrong:
                self.done = True
                return self._state(), -50.0, True
            # Escalating penalty for each wrong guess
            penalty = -2.0 - (self.wrong * 1.5)
            return self._state(), penalty, False

# --- DQN Agent ---
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

# Helper to build state vector for agent
def build_state_vector(masked_word: str, guessed_letters: set, hprob: np.ndarray, max_len: Optional[int] = None) -> np.ndarray:
    """
    Build a fixed-size state vector. If max_len is provided, pad (with '_') or truncate
    the masked_word to that length so the resulting vector is always the same size.
    """
    # masked_word: string of length L with letters or '_'
    # guessed_letters: set of chars
    # hprob: 26-dim vector from oracle
    if max_len is None:
        L = len(masked_word)
    else:
        L = max_len
    # normalize/pad/truncate masked_word to length L
    if len(masked_word) < L:
        mw = masked_word + ('_' * (L - len(masked_word)))
    else:
        mw = masked_word[:L]
    # one-hot each position for 27 symbols (a-z + blank)
    pos_vec = []
    for ch in mw:
        v = np.zeros(27, dtype=np.float32)
        if ch == '_':
            v[26] = 1.0
        else:
            # safety for non-alpha
            if 'a' <= ch <= 'z':
                v[ord(ch) - 97] = 1.0
            else:
                v[26] = 1.0
        pos_vec.append(v)
    pos_vec = np.concatenate(pos_vec)
    # guessed vector
    guessed_vec = np.zeros(26, dtype=np.float32)
    for g in guessed_letters:
        if 'a' <= g <= 'z':
            guessed_vec[ord(g)-97] = 1.0
    # combine: pos_vec | guessed_vec | hprob
    st = np.concatenate([pos_vec, guessed_vec, hprob.astype(np.float32)])
    return st

# Action masking when selecting letter: set q-values for guessed letters to very low

def select_action(policy_net: DQN, state_vec: np.ndarray, guessed_letters: set, eps: float, device: torch.device, 
                  hprob: Optional[np.ndarray] = None, masked_word: Optional[str] = None):
    """
    Enhanced action selection with multiple strategies:
    1. First 6 guesses: use strategic frequency (e, a, i, o, t, n)
    2. Early exploration: occasionally use frequency-based selection
    3. Main strategy: heavily weight oracle (90%) with RL (10%)
    4. Fallback: if oracle uncertain, use frequency
    """
    
    # Strategy 1: Strategic first guesses
    if masked_word:
        first_guess = get_smart_first_guesses(masked_word, guessed_letters)
        if first_guess and first_guess not in guessed_letters:
            return first_guess
    
    # Strategy 2: Exploration - use frequency-based
    if random.random() < eps:
        # Use frequency-based selection, not random
        choices = [c for c in LETTER_FREQ if c not in guessed_letters]
        if choices:
            return choices[0]
        # Fallback to random
        choices = [c for c in list('abcdefghijklmnopqrstuvwxyz') if c not in guessed_letters]
        if not choices:
            return random.choice(list('abcdefghijklmnopqrstuvwxyz'))
        return random.choice(choices)
    
    # Strategy 3: Trust oracle heavily, use RL as minor adjustment
    if hprob is not None:
        state_t = torch.from_numpy(state_vec).unsqueeze(0).to(device)
        with torch.no_grad():
            q = policy_net(state_t).cpu().numpy().squeeze()
        
        # Normalize q-values to 0-1 range
        q_min, q_max = q.min(), q.max()
        if q_max > q_min:
            q_norm = (q - q_min) / (q_max - q_min)
        else:
            q_norm = np.ones_like(q) / 26

        # Blend: 5% RL, 95% Oracle
        combined = 0.15 * q_norm + 0.85 * hprob
        
        # Check if oracle is uncertain (entropy high)
        entropy = -np.sum(hprob * np.log(hprob + 1e-10))
        max_entropy = np.log(26)
        
        # If oracle is very uncertain, fall back to frequency
        if entropy > 0.8 * max_entropy:
            # Use frequency as fallback
            freq_probs = np.zeros(26)
            for i, letter in enumerate(LETTER_FREQ):
                freq_probs[ord(letter) - 97] = (26 - i) / sum(range(1, 27))
            combined = 0.5 * combined + 0.5 * freq_probs
        
        # Mask guessed letters
        for g in guessed_letters:
            if 'a' <= g <= 'z':
                combined[ord(g)-97] = -1e9
        
        # Choose best
        idx = int(np.argmax(combined))
        return chr(idx + 97)
    
    # Fallback: pure RL
    state_t = torch.from_numpy(state_vec).unsqueeze(0).to(device)
    with torch.no_grad():
        q = policy_net(state_t).cpu().numpy().squeeze()
    # mask guessed letters
    for g in guessed_letters:
        if 'a' <= g <= 'z':
            q[ord(g)-97] = -1e9
    # choose argmax
    idx = int(np.argmax(q))
    return chr(idx + 97)

# Training function

def train_dqn(corpus_words: List[str], oracle: LengthSpecificNGramOracle, test_words: List[str],
              num_episodes: int = 2000, batch_size: int = 128, max_wrong: int = 6,
              device: Optional[torch.device] = None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # We'll restrict training to word lengths that oracle supports
    lens = sorted(oracle.length_models.keys())
    if not lens:
        raise ValueError('Oracle has no trained length models; increase corpus or reduce min_examples')
    # Prepare sizes
    # Use the maximum supported length so all state vectors (padded) match the network
    rep_len = max(lens)
    input_dim = rep_len * 27 + 26 + 26
    hidden_dim = 256
    output_dim = 26
    policy_net = DQN(input_dim, hidden_dim, output_dim).to(device)
    target_net = DQN(input_dim, hidden_dim, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=5e-4)  # Lower learning rate
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    replay = ReplayBuffer(20000)

    gamma = 0.99
    eps_start = 0.3  # Lower exploration since we trust oracle
    eps_end = 0.01
    eps_decay = num_episodes * 0.5
    target_update = 50

    episode_rewards = []

    for ep in range(1, num_episodes+1):
        # sample a random word from corpus with supported length
        while True:
            w = random.choice(corpus_words)
            if len(w) in oracle.length_models:
                break
        env = HangmanEnv(w, max_wrong=max_wrong)
        masked, guessed, lives = env.reset()
        total_reward = 0.0
        done = False
        step_count = 0
        while not done:
            # oracle probability
            hprob = oracle.predict_mask_prob(masked, guessed)
            state_vec = build_state_vector(masked, guessed, hprob, max_len=rep_len)
            eps = eps_end + (eps_start - eps_end) * math.exp(-1. * ep / eps_decay)
            act = select_action(policy_net, state_vec, guessed, eps, device, hprob=hprob, masked_word=masked)
            next_state, reward, done = env.step(act)
            masked2, guessed2, lives2 = next_state
            hprob2 = oracle.predict_mask_prob(masked2, guessed2)
            next_state_vec = build_state_vector(masked2, guessed2, hprob2, max_len=rep_len)
            replay.push((state_vec, ord(act)-97, reward, next_state_vec, done))
            total_reward += reward
            step_count += 1
            # learning step
            if len(replay) >= batch_size:
                batch = replay.sample(batch_size)
                states = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
                actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
                rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
                next_states = torch.from_numpy(np.stack([b[3] for b in batch])).to(device)
                dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device)

                q_values = policy_net(states)
                q_val = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                expected = rewards + gamma * next_q * (1 - dones)
                loss = nn.functional.mse_loss(q_val, expected)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            masked, guessed = masked2, guessed2
        episode_rewards.append(total_reward)
        if ep % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        scheduler.step()  # Update learning rate
        if ep % 100 == 0:
            avg_r = np.mean(episode_rewards[-100:])
            print(f"Episode {ep}/{num_episodes} avg reward (last100): {avg_r:.2f}")
    # Save policy
    # torch.save(policy_net.state_dict(), 'dqn_policy.pth')
    return policy_net, episode_rewards

# Evaluation

def evaluate_agent(policy_net: DQN, oracle: LengthSpecificNGramOracle, test_words: List[str], max_wrong: int = 6, device: Optional[torch.device]=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_net.to(device)
    policy_net.eval()
    success = 0
    total_wrong = 0
    total_repeats = 0
    games = 0
    # ensure we use same fixed input size as during training
    if not oracle.length_models:
        raise ValueError("Oracle has no length models.")
    eval_rep_len = max(oracle.length_models.keys())

    print(f"[DEBUG] Starting evaluation on {len(test_words)} games. eval_rep_len={eval_rep_len}")

    for i, w in enumerate(test_words, start=1):
        print(f"[DEBUG] Game {i}/{len(test_words)} start. word='{w}'")
        env = HangmanEnv(w, max_wrong=max_wrong)
        masked, guessed, lives = env.reset()
        done = False
        per_game_repeats_before = total_repeats
        steps = 0
        while not done:
            hprob = oracle.predict_mask_prob(masked, guessed)
            state_vec = build_state_vector(masked, guessed, hprob, max_len=eval_rep_len)
            act = select_action(policy_net, state_vec, guessed, eps=0.0, device=device, hprob=hprob, masked_word=masked)
            print(f"[DEBUG]  Step {steps} masked='{masked}' guessed={sorted(list(guessed))} -> act='{act}'")
            # check repeated
            if act in guessed:
                total_repeats += 1
                print(f"[DEBUG]   Repeated guess detected for '{act}' (total_repeats={total_repeats})")
            prev_wrong = env.wrong
            _, reward, done = env.step(act)
            if env.wrong > prev_wrong:
                total_wrong += 1
                print(f"[DEBUG]   Wrong guess. wrong count now {env.wrong} (total_wrong={total_wrong})")
            # update masked/guessed for next loop
            masked, guessed = env._state()[0], env._state()[1]
            steps += 1
        per_game_repeats = total_repeats - per_game_repeats_before
        outcome = 'WIN' if '_' not in env.mask else 'LOSE'
        print(f"[DEBUG] Game {i} finished: {outcome} steps={steps} wrong={env.wrong} repeats={per_game_repeats}")
        if '_' not in env.mask:
            success += 1
        games += 1

    success_rate = success / games if games > 0 else 0.0
    print(f"[DEBUG] Evaluation complete. games={games} success={success} success_rate={success_rate:.3f} total_wrong={total_wrong} total_repeats={total_repeats}")

    return {
        'success_rate': success_rate,
        'total_wrong': total_wrong,
        'total_repeats': total_repeats,
        'games': games
    }

# --- Main orchestration (if run as script) ---
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    data_dir = 'Data'
    corpus_path = os.path.join(data_dir, 'corpus.txt')
    test_path = os.path.join(data_dir, 'test.txt')
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")
    corpus_words = load_wordlist(corpus_path)
    test_words = []
    if os.path.exists(test_path):
        test_words = load_wordlist(test_path)
    else:
        # if no test.txt provided, use a held-out subset
        random.shuffle(corpus_words)
        split = int(0.9 * len(corpus_words))
        test_words = corpus_words[split:split+1000]
        corpus_words = corpus_words[:split]

    # Train oracle
    oracle = LengthSpecificNGramOracle(n=2, min_examples=30)
    oracle.fit(corpus_words)

    # Train DQN (longer training for better convergence)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_net, rewards = train_dqn(corpus_words, oracle, test_words, num_episodes=3000, batch_size=128, max_wrong=6, device=device)

    # Plot training rewards
    plt.figure(figsize=(8,4))
    plt.plot(rewards)
    plt.title('Episode rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.tight_layout()
    # plt.savefig('training_rewards.png')
    plt.show()

    # Evaluate on test set (use 1000 games as spec suggests)
    eval_sample = test_words
    results = evaluate_agent(policy_net, oracle, eval_sample, max_wrong=6, device=device)
    final_score = results['success_rate'] * 2000 - (results['total_wrong'] * 5) - (results['total_repeats'] * 2)
    print('Evaluation results:')
    print(json.dumps(results, indent=2))
    print(f'Final Score (per hackathon formula): {final_score:.2f}')

    # Save oracle for inspection
    # with open('oracle_summary.json', 'w') as f:
    #     json.dump({'lengths': list(oracle.length_models.keys())}, f)

    print('Done.')
