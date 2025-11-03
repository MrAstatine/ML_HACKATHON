# 🧠 ML Hackathon – Hybrid HMM + RL Hangman Solver

## Overview
This project builds an **intelligent Hangman-playing agent** that blends **Hidden Markov Models (HMM)** with **Reinforcement Learning (RL)** to predict letters efficiently and minimize wrong guesses.

It demonstrates how statistical sequence modeling (HMM) and learning-based decision-making (RL) can be combined to improve word-guessing accuracy — aligning with the hackathon theme of *explainable, interpretable, and high-performing AI systems.*

---

## Approach summary
Component descriptions:

- HMM — Learns letter transition probabilities per word length (4–17). Provides base probabilities for likely next letters.
- RL Agent (DQN) — Learns optimal letter-guessing strategy from gameplay rewards using Q-Learning.
- Hybrid System — Weighted combination of both models: `Final_Score = α * RL + (1-α) * HMM`, tuned between 0.85–0.95 RL.
- Explainability — Logs per-letter probabilities and Q-values / reward signals to visualize why each letter is chosen.

---

## Project structure
Top-level folders in this repository:

- `code/` — All training and tuning scripts
- `aaron-ml-review-1/` — Data, models, and Jupyter notebooks
- `docs/` — Improvement logs and tuning notes
- `results/` — Saved evaluation and tuning outputs
- `Data/` — `test.txt` for evaluation

---

## Setup & run

### Prerequisites
- Python ≥ 3.11
- Install dependencies:

```powershell
pip install -r requirements.txt
```

### Train or evaluate models
Run the main scripts from the `code/` folder:

```powershell
# Run hybrid training (HMM + RL)
python code/90HMM_10RL_0.2685.py

# Tune weights between RL and HMM
python code/tune_rl_hmm_split.py

# Evaluate final model
python code/test.py
```

---

## Key results
Metric | Baseline | Improved
---|---:|---:
Success Rate | 1.2% | 50–70%
Avg. Wrong Guesses | 11.96 | 3.5–5.0
Final Score | −59,791 | +30,000 to +60,000

Interpretability: The model records HMM probabilities and Q-values per prediction for analysis and visualization in notebooks under `aaron-ml-review-1/notebooks/`.

---

## Highlights
- Hybrid AI using HMM + RL for sequential decision-making
- Data-driven explainability with interpretable probability outputs
- Automated hyperparameter tuning for both Q-learning and weighting
- Significant improvement in accuracy and efficiency over baseline models

## Future work
- Add LIME/SHAP explainers for deeper interpretability
- Deploy as an interactive web app or leaderboard bot
- Expand to multi-language word datasets

---

## Authors
Primary author and contributors:

- Akshat Tripathi - https://github.com/MrAstatine
- Advaith Sanil Kumar — https://github.com/askadvaith
- Aaron Sabu — https://github.com/aaron-sabu07
- Aashlesh Lokesh — https://github.com/aashlesh-lokesh

---

## Notes
The project demonstrates a hybrid HMM + RL approach with explainability hooks and automated tuning. See `code/` and the notebooks under `aaron-ml-review-1/notebooks/` for implementation and evaluation details.
