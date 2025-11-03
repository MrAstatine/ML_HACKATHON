# Hyperparameter Tuning Scripts

This directory contains two automated tuning scripts to optimize the Hangman agent.

## Scripts Overview

### 1. `tune_rl_hmm_split.py` - RL/HMM Weight Exploration

**Purpose:** Find the optimal balance between RL and HMM predictions.

**What it tests:**
- 11 different weight combinations from Pure HMM (0/1) to Pure RL (1/0)
- Tests: 0.0/1.0, 0.1/0.9, 0.2/0.8, 0.3/0.7, 0.4/0.6, 0.5/0.5, 0.6/0.4, 0.7/0.3, 0.8/0.2, 0.9/0.1, 1.0/0.0

**Configuration:**
- Training: 2000 episodes per configuration
- Evaluation: 500 test games per configuration
- Total runtime: ~2-3 hours (depends on GPU/CPU)

**Output:**
- Console: Live progress and summary table
- File: `rl_hmm_split_results.json` - Full results with best configuration

**Usage:**
```powershell
cd code
python tune_rl_hmm_split.py
```

**Expected Output:**
```
RL/HMM WEIGHT EXPLORATION
================================================================================
...
SUMMARY - ALL RESULTS
================================================================================
    RL    HMM    Success     Wrong  Repeats      Score Train Reward
--------------------------------------------------------------------------------
   0.0    1.0      45.2%      1234       12      850.42        12.34
   0.1    0.9      52.6%       982        8     1023.16        15.67
   ...
BEST CONFIGURATION
================================================================================
RL Weight: 0.1
HMM Weight: 0.9
Success Rate: 52.6%
Final Score: 1023.16
```

---

### 2. `tune_hyperparameters.py` - Hyperparameter Optimization

**Purpose:** Find the best hyperparameters for the DQN network.

**What it tests (Random Search):**
- `dropout`: [0.1, 0.2, 0.3, 0.4]
- `learning_rate`: [1e-4, 5e-4, 1e-3]
- `hidden_dim`: [128, 256, 512]
- `eps_start`: [0.2, 0.3, 0.5]
- `min_examples`: [20, 30, 40] (oracle training threshold)

**Configuration:**
- Configurations tested: 15 random combinations
- Training: 2000 episodes per configuration
- Evaluation: 500 test games per configuration
- Total runtime: ~3-4 hours (depends on GPU/CPU)

**Output:**
- Console: Live progress and top 5 configurations
- File: `hyperparameter_tuning_results.json` - Full results ranked by score

**Usage:**
```powershell
cd code
python tune_hyperparameters.py
```

**Expected Output:**
```
HYPERPARAMETER TUNING
================================================================================
Testing 15 random configurations

Configuration 1/15
================================================================================
Hyperparameters:
  dropout: 0.3
  learning_rate: 0.0005
  hidden_dim: 256
  eps_start: 0.3
  min_examples: 30
...
SUMMARY - TOP 5 CONFIGURATIONS
================================================================================

1. Score: 1045.32 | Success: 54.2%
   Config: {'dropout': 0.3, 'learning_rate': 0.0005, 'hidden_dim': 256, ...}
   Wrong: 945, Repeats: 7
...
```

---

## Tips for Running

### Speed Up Tuning (if needed)

Edit the constants at the top of each script:

**For faster exploration:**
```python
TRAINING_EPISODES = 1000  # Reduce from 2000
EVAL_SAMPLE_SIZE = 250    # Reduce from 500
NUM_RANDOM_CONFIGS = 10   # Reduce from 15 (hyperparameters only)
```

**For more thorough search:**
```python
TRAINING_EPISODES = 3000  # Increase from 2000
EVAL_SAMPLE_SIZE = 1000   # Increase from 500
NUM_RANDOM_CONFIGS = 25   # Increase from 15 (hyperparameters only)
```

### Recommended Workflow

1. **First run `tune_rl_hmm_split.py`**
   - This finds the best RL/HMM balance
   - Relatively fast (~2-3 hours)
   - Results are stable across runs

2. **Update `tune_hyperparameters.py` with best split**
   - Open `tune_hyperparameters.py`
   - Update lines 23-24 with best weights from step 1:
     ```python
     RL_WEIGHT = 0.1   # Use best from script 1
     HMM_WEIGHT = 0.9
     ```

3. **Run `tune_hyperparameters.py`**
   - This optimizes network architecture
   - Takes longer (~3-4 hours)
   - Can run overnight

4. **Apply best configuration to main script**
   - Use the best hyperparameters in `gpt_trial1.py`
   - Update the relevant sections with winning values

### GPU Acceleration

Both scripts automatically use GPU if available (CUDA). Check with:
```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If you have a GPU, expect 3-5x speedup.

---

## Output Files

### `rl_hmm_split_results.json`
```json
{
  "results": [
    {
      "rl_weight": 0.1,
      "hmm_weight": 0.9,
      "success_rate": 0.526,
      "total_wrong": 982,
      "total_repeats": 8,
      "final_score": 1023.16,
      "avg_training_reward": 15.67
    },
    ...
  ],
  "best": { ... },
  "config": { ... }
}
```

### `hyperparameter_tuning_results.json`
```json
{
  "results": [
    {
      "config": {
        "dropout": 0.3,
        "learning_rate": 0.0005,
        "hidden_dim": 256,
        "eps_start": 0.3,
        "min_examples": 30
      },
      "success_rate": 0.542,
      "total_wrong": 945,
      "total_repeats": 7,
      "final_score": 1045.32,
      "avg_training_reward": 16.23
    },
    ...
  ],
  "best": { ... },
  "config": { ... }
}
```

---

## Interpreting Results

**Key Metrics:**
- **Success Rate**: % of games won (most important)
- **Final Score**: `(success_rate × 2000) - (wrong × 5) - (repeats × 2)`
- **Total Wrong**: Number of incorrect guesses across all games
- **Total Repeats**: Number of repeated guesses (should be low)
- **Avg Training Reward**: Convergence indicator (higher = better learning)

**Good Results:**
- Success Rate: > 40%
- Final Score: > 500
- Total Repeats: < 20

**Excellent Results:**
- Success Rate: > 50%
- Final Score: > 800
- Total Repeats: < 10

---

## Troubleshooting

**Out of Memory (GPU):**
- Reduce `EVAL_SAMPLE_SIZE`
- Reduce `hidden_dim` in hyperparameter grid
- Close other GPU applications

**Too Slow:**
- Reduce `TRAINING_EPISODES`
- Reduce `NUM_RANDOM_CONFIGS`
- Use GPU instead of CPU

**Poor Results:**
- Increase `TRAINING_EPISODES`
- Expand hyperparameter grid
- Check data files (corpus.txt, test.txt)

---

## Next Steps After Tuning

1. Apply best RL/HMM weights to `gpt_trial1.py`
2. Apply best hyperparameters to `gpt_trial1.py`
3. Run full training with optimized settings (5000+ episodes)
4. Evaluate on full test set (2000 games)
5. Submit to competition!
