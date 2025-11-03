# Script 2: Hyperparameter Tuning
# This script explores different hyperparameter combinations to find optimal settings

import os
import sys
import random
import math
import json
import itertools
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- Hyperparameter Grid ---
HYPERPARAMETER_GRID = {
    'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
    'learning_rate': [5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
    'hidden_dim': [64, 128, 192, 256, 384, 512],
    'eps_start': [0.1, 0.2, 0.3, 0.4, 0.5],
    'eps_decay_factor': [0.3, 0.4, 0.5, 0.6, 0.7],  # Multiplied by num_episodes
    'min_examples': [15, 20, 25, 30, 35, 40, 50],
    'batch_size': [32, 64, 128, 256],
    'replay_buffer_size': [10000, 20000, 30000, 50000],
    'target_update_freq': [25, 50, 75, 100],
    'gamma': [0.95, 0.97, 0.99, 0.995],
    'scheduler_step_size': [300, 500, 700, 1000],
    'scheduler_gamma': [0.85, 0.9, 0.95],
    'smoothing': [1.0, 1.5, 2.0, 2.5, 3.0],  # HMM smoothing parameter
}

# Fixed parameters
TRAINING_EPISODES = 2000
EVAL_SAMPLE_SIZE = 500
RL_WEIGHT = 0.1  # Use best from script 1 - DO NOT CHANGE
HMM_WEIGHT = 0.9  # DO NOT CHANGE

# For faster search, we'll do random search instead of full grid
NUM_RANDOM_CONFIGS = 40  # Test 40 random configurations

# --- Utility functions ---

def load_wordlist(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        words = [w.strip().lower() for w in f if w.strip()]
    return words

LETTER_FREQ = ['e', 'a', 'i', 'o', 't', 'n', 's', 'r', 'h', 'l', 'd', 'c', 'u', 'm', 'p', 'f', 'g', 'w', 'y', 'b', 'v', 'k', 'x', 'j', 'q', 'z']

def get_smart_first_guesses(masked_word: str, guessed: set) -> Optional[str]:
    if len(guessed) == 0: return 'e'
    if len(guessed) == 1: return 'a'
    if len(guessed) == 2: return 'i'
    if len(guessed) == 3: return 'o'
    if len(guessed) == 4: return 't'
    if len(guessed) == 5: return 'n'
    return None

# --- HMM Oracle ---
class LengthSpecificNGramOracle:
    def __init__(self, n=2, min_examples=30, smoothing=2.0):
        self.n = n
        self.min_examples = min_examples
        self.smoothing = smoothing
        self.alphabet = list("abcdefghijklmnopqrstuvwxyz")
        self.idx = {c:i for i,c in enumerate(self.alphabet)}
        self.length_models = {}
        self.length_totals = {}

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
            totals = {ctx: sum(cnts.values()) for ctx, cnts in models.items()}
            self.length_totals[L] = totals

    def _prob_next_char(self, ctx: str, length: int) -> np.ndarray:
        vocab = self.alphabet + ["$"]
        V = len(vocab)
        probs = np.ones(len(vocab)) * self.smoothing
        if length in self.length_models and ctx in self.length_models[length]:
            counts = self.length_models[length][ctx]
            for i, ch in enumerate(vocab):
                probs[i] += counts.get(ch, 0)
            probs = probs / probs.sum()
        else:
            probs = probs / probs.sum()
        return probs[:-1]

    def predict_mask_prob(self, masked_word: str, guessed_letters: set) -> np.ndarray:
        L = len(masked_word)
        if L not in self.length_models:
            if not self.length_models:
                return np.ones(26) / 26
            lengths = sorted(self.length_models.keys(), key=lambda x: abs(x-L))
            L = lengths[0]
        
        probs_accum = np.zeros(26)
        blanks = 0
        padded = ("^" * (self.n-1)) + masked_word + "$"
        for i in range(self.n-1, self.n-1+len(masked_word)):
            ch = padded[i]
            if ch == '_':
                blanks += 1
                ctx = padded[i-(self.n-1):i]
                ctx_clean = ''.join([c if c != '_' else '^' for c in ctx])
                pnext = self._prob_next_char(ctx_clean, L)
                probs_accum += pnext
        if blanks == 0:
            return np.zeros(26)
        probs = probs_accum / blanks
        
        # Pattern boosts
        if 'q' in masked_word and 'u' not in guessed_letters:
            probs[self.idx['u']] *= 20.0
        if masked_word.endswith('_') and len(masked_word) > 2:
            for letter, boost in [('e', 3.0), ('s', 2.5), ('d', 2.0), ('t', 1.8), ('y', 1.5)]:
                if letter not in guessed_letters:
                    probs[self.idx[letter]] *= boost
        if 'ng' in masked_word or '_n' in masked_word or 'n_' in masked_word:
            if 'i' not in guessed_letters:
                probs[self.idx['i']] *= 3.0
        for letter in set(masked_word) - {'_'}:
            if letter in self.idx and letter not in guessed_letters:
                probs[self.idx[letter]] *= 1.5
        if 'th' in masked_word and 'e' not in guessed_letters:
            probs[self.idx['e']] *= 2.0
        
        for g in guessed_letters:
            if g in self.idx:
                probs[self.idx[g]] = 0.0
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
        letter = letter.lower()
        if self.done:
            raise RuntimeError('Game already finished')
        if letter in self.guessed:
            reward = -2.0
            done = False
            return self._state(), reward, self.done
        self.guessed.add(letter)
        if letter in self.word:
            num_revealed = 0
            for i, ch in enumerate(self.word):
                if ch == letter:
                    self.mask[i] = letter
                    num_revealed += 1
            if '_' not in self.mask:
                self.done = True
                return self._state(), 50.0 + (self.max_wrong - self.wrong) * 5.0, True
            return self._state(), 2.0 + num_revealed, False
        else:
            self.wrong += 1
            if self.wrong >= self.max_wrong:
                self.done = True
                return self._state(), -50.0, True
            penalty = -2.0 - (self.wrong * 1.5)
            return self._state(), penalty, False

# --- DQN Agent ---
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
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

def build_state_vector(masked_word: str, guessed_letters: set, hprob: np.ndarray, max_len: Optional[int] = None) -> np.ndarray:
    if max_len is None:
        L = len(masked_word)
    else:
        L = max_len
    if len(masked_word) < L:
        mw = masked_word + ('_' * (L - len(masked_word)))
    else:
        mw = masked_word[:L]
    pos_vec = []
    for ch in mw:
        v = np.zeros(27, dtype=np.float32)
        if ch == '_':
            v[26] = 1.0
        else:
            if 'a' <= ch <= 'z':
                v[ord(ch) - 97] = 1.0
            else:
                v[26] = 1.0
        pos_vec.append(v)
    pos_vec = np.concatenate(pos_vec)
    guessed_vec = np.zeros(26, dtype=np.float32)
    for g in guessed_letters:
        if 'a' <= g <= 'z':
            guessed_vec[ord(g)-97] = 1.0
    st = np.concatenate([pos_vec, guessed_vec, hprob.astype(np.float32)])
    return st

def select_action(policy_net: DQN, state_vec: np.ndarray, guessed_letters: set, eps: float, device: torch.device, 
                  hprob: Optional[np.ndarray] = None, masked_word: Optional[str] = None):
    
    if masked_word:
        first_guess = get_smart_first_guesses(masked_word, guessed_letters)
        if first_guess and first_guess not in guessed_letters:
            return first_guess
    
    if random.random() < eps:
        choices = [c for c in LETTER_FREQ if c not in guessed_letters]
        if choices:
            return choices[0]
        choices = [c for c in list('abcdefghijklmnopqrstuvwxyz') if c not in guessed_letters]
        if not choices:
            return random.choice(list('abcdefghijklmnopqrstuvwxyz'))
        return random.choice(choices)
    
    if hprob is not None:
        state_t = torch.from_numpy(state_vec).unsqueeze(0).to(device)
        with torch.no_grad():
            q = policy_net(state_t).cpu().numpy().squeeze()
        
        q_min, q_max = q.min(), q.max()
        if q_max > q_min:
            q_norm = (q - q_min) / (q_max - q_min)
        else:
            q_norm = np.ones_like(q) / 26
        
        combined = RL_WEIGHT * q_norm + HMM_WEIGHT * hprob
        
        entropy = -np.sum(hprob * np.log(hprob + 1e-10))
        max_entropy = np.log(26)
        
        if entropy > 0.8 * max_entropy:
            freq_probs = np.zeros(26)
            for i, letter in enumerate(LETTER_FREQ):
                freq_probs[ord(letter) - 97] = (26 - i) / sum(range(1, 27))
            combined = 0.5 * combined + 0.5 * freq_probs
        
        for g in guessed_letters:
            if 'a' <= g <= 'z':
                combined[ord(g)-97] = -1e9
        
        idx = int(np.argmax(combined))
        return chr(idx + 97)
    
    state_t = torch.from_numpy(state_vec).unsqueeze(0).to(device)
    with torch.no_grad():
        q = policy_net(state_t).cpu().numpy().squeeze()
    for g in guessed_letters:
        if 'a' <= g <= 'z':
            q[ord(g)-97] = -1e9
    idx = int(np.argmax(q))
    return chr(idx + 97)

def train_dqn(corpus_words: List[str], oracle: LengthSpecificNGramOracle, test_words: List[str],
              num_episodes: int, batch_size: int, max_wrong: int, device: torch.device,
              dropout: float, learning_rate: float, hidden_dim: int, eps_start: float,
              eps_decay_factor: float, replay_buffer_size: int, target_update_freq: int,
              gamma: float, scheduler_step_size: int, scheduler_gamma: float):
    """Training with configurable hyperparameters"""
    
    lens = sorted(oracle.length_models.keys())
    if not lens:
        raise ValueError('Oracle has no trained length models')
    rep_len = max(lens)
    input_dim = rep_len * 27 + 26 + 26
    output_dim = 26
    
    policy_net = DQN(input_dim, hidden_dim, output_dim, dropout).to(device)
    target_net = DQN(input_dim, hidden_dim, output_dim, dropout).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    replay = ReplayBuffer(replay_buffer_size)

    eps_end = 0.01
    eps_decay = num_episodes * eps_decay_factor
    target_update = target_update_freq

    episode_rewards = []

    for ep in range(1, num_episodes+1):
        while True:
            w = random.choice(corpus_words)
            if len(w) in oracle.length_models:
                break
        env = HangmanEnv(w, max_wrong=max_wrong)
        masked, guessed, lives = env.reset()
        total_reward = 0.0
        done = False
        while not done:
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
        scheduler.step()
    return policy_net, episode_rewards

def evaluate_agent(policy_net: DQN, oracle: LengthSpecificNGramOracle, test_words: List[str], 
                   max_wrong: int, device: torch.device):
    policy_net.to(device)
    policy_net.eval()
    success = 0
    total_wrong = 0
    total_repeats = 0
    games = 0
    if not oracle.length_models:
        raise ValueError("Oracle has no length models.")
    eval_rep_len = max(oracle.length_models.keys())

    for i, w in enumerate(test_words, start=1):
        env = HangmanEnv(w, max_wrong=max_wrong)
        masked, guessed, lives = env.reset()
        done = False
        while not done:
            hprob = oracle.predict_mask_prob(masked, guessed)
            state_vec = build_state_vector(masked, guessed, hprob, max_len=eval_rep_len)
            act = select_action(policy_net, state_vec, guessed, eps=0.0, device=device, hprob=hprob, masked_word=masked)
            if act in guessed:
                total_repeats += 1
            prev_wrong = env.wrong
            _, reward, done = env.step(act)
            if env.wrong > prev_wrong:
                total_wrong += 1
            masked, guessed = env._state()[0], env._state()[1]
        if '_' not in env.mask:
            success += 1
        games += 1

    success_rate = success / games if games > 0 else 0.0
    return {
        'success_rate': success_rate,
        'total_wrong': total_wrong,
        'total_repeats': total_repeats,
        'games': games
    }

def generate_random_configs(grid, num_configs):
    """Generate random configurations from hyperparameter grid"""
    configs = []
    keys = list(grid.keys())
    for _ in range(num_configs):
        config = {key: random.choice(grid[key]) for key in keys}
        configs.append(config)
    return configs

# --- Main Execution ---
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    data_dir = 'Data'
    corpus_path = os.path.join(data_dir, 'corpus.txt')
    test_path = os.path.join(data_dir, 'test.txt')
    
    print("="*80)
    print("HYPERPARAMETER TUNING")
    print("="*80)
    
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")
    
    corpus_words = load_wordlist(corpus_path)
    test_words = []
    if os.path.exists(test_path):
        test_words = load_wordlist(test_path)
    else:
        random.shuffle(corpus_words)
        split = int(0.9 * len(corpus_words))
        test_words = corpus_words[split:split+1000]
        corpus_words = corpus_words[:split]
    
    eval_sample = random.sample(test_words, min(EVAL_SAMPLE_SIZE, len(test_words)))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Testing {NUM_RANDOM_CONFIGS} random configurations\n")
    
    # Generate random configs
    configs = generate_random_configs(HYPERPARAMETER_GRID, NUM_RANDOM_CONFIGS)
    
    results = []
    
    for idx, config in enumerate(configs, 1):
        print(f"\n{'='*80}")
        print(f"Configuration {idx}/{NUM_RANDOM_CONFIGS}")
        print(f"{'='*80}")
        print(f"Hyperparameters:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Train oracle with config min_examples and smoothing
        oracle = LengthSpecificNGramOracle(
            n=2, 
            min_examples=config['min_examples'],
            smoothing=config['smoothing']
        )
        oracle.fit(corpus_words)
        
        # Train
        print(f"\nTraining for {TRAINING_EPISODES} episodes...")
        policy_net, rewards = train_dqn(
            corpus_words, oracle, test_words, 
            num_episodes=TRAINING_EPISODES, 
            batch_size=config['batch_size'], 
            max_wrong=6, 
            device=device,
            dropout=config['dropout'],
            learning_rate=config['learning_rate'],
            hidden_dim=config['hidden_dim'],
            eps_start=config['eps_start'],
            eps_decay_factor=config['eps_decay_factor'],
            replay_buffer_size=config['replay_buffer_size'],
            target_update_freq=config['target_update_freq'],
            gamma=config['gamma'],
            scheduler_step_size=config['scheduler_step_size'],
            scheduler_gamma=config['scheduler_gamma']
        )
        
        avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        print(f"Avg training reward (last 100): {avg_reward:.2f}")
        
        # Evaluate
        print(f"Evaluating on {len(eval_sample)} test games...")
        eval_results = evaluate_agent(policy_net, oracle, eval_sample, max_wrong=6, device=device)
        
        final_score = (eval_results['success_rate'] * 2000 - 
                      eval_results['total_wrong'] * 5 - 
                      eval_results['total_repeats'] * 2)
        
        result = {
            'config': config,
            'success_rate': eval_results['success_rate'],
            'total_wrong': eval_results['total_wrong'],
            'total_repeats': eval_results['total_repeats'],
            'final_score': final_score,
            'avg_training_reward': avg_reward
        }
        results.append(result)
        
        print(f"\nResults:")
        print(f"  Success Rate: {eval_results['success_rate']:.1%}")
        print(f"  Total Wrong: {eval_results['total_wrong']}")
        print(f"  Total Repeats: {eval_results['total_repeats']}")
        print(f"  Final Score: {final_score:.2f}")
    
    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY - TOP 5 CONFIGURATIONS")
    print(f"{'='*80}\n")
    
    # Sort by score
    results.sort(key=lambda x: x['final_score'], reverse=True)
    
    for idx, r in enumerate(results[:5], 1):
        print(f"\n{idx}. Score: {r['final_score']:.2f} | Success: {r['success_rate']:.1%}")
        print(f"   Config: {r['config']}")
        print(f"   Wrong: {r['total_wrong']}, Repeats: {r['total_repeats']}")
    
    # Best
    best = results[0]
    print(f"\n{'='*80}")
    print("BEST CONFIGURATION")
    print(f"{'='*80}")
    print(f"Final Score: {best['final_score']:.2f}")
    print(f"Success Rate: {best['success_rate']:.1%}")
    print(f"Hyperparameters:")
    for key, value in best['config'].items():
        print(f"  {key}: {value}")
    
    # Save results
    os.makedirs('docs', exist_ok=True)
    output_file = 'docs/hyperparameter_tuning_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'best': best,
            'config': {
                'training_episodes': TRAINING_EPISODES,
                'eval_sample_size': EVAL_SAMPLE_SIZE,
                'num_configs_tested': NUM_RANDOM_CONFIGS,
                'rl_weight': RL_WEIGHT,
                'hmm_weight': HMM_WEIGHT
            }
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")
