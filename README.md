# Reinforcement Learning Agent for 2048

An academic implementation of classical Reinforcement Learning techniques for mastering the game **2048**, achieving tiles up to **8192** using efficient value-based methods — **without deep neural networks**.

This project demonstrates how careful problem formulation, feature engineering, and afterstate learning enable strong performance in a highly stochastic, large state-space environment.

---

## 🎮 Project Overview

The goal of this project is to train an autonomous agent to play the game 2048 using **Afterstate Value Learning**, **Temporal Difference (TD) learning**, and **N-tuple feature representations with symmetry exploitation**.

Despite the enormous theoretical state space (~10²⁴), the agent achieves competitive performance through:
- Compact state encoding
- Local pattern-based value approximation
- Variance reduction via afterstate evaluation

### Key Results
- Maximum tile achieved: **8192**
- Best average score: **62,704**
- **72.4%** success rate reaching 2048
- **68.4%** success rate reaching 4096
- Trained using classical RL methods only (no deep learning)

---

## 🧠 Methodology

### Markov Decision Process (MDP)

**State Space**  
- 4×4 board, tiles encoded as `log₂(value)`  
- Values range from 0 (empty) to 13 (8192)  
- Logarithmic encoding enables efficient indexing and pattern matching

**Action Space**  
- Four actions: Up, Down, Left, Right  
- Invalid moves (no board change) are filtered

**Reward Function**  
- Reward equals the value of merged tiles  
- Example: merging 2 + 2 → reward = 4

**Environment Dynamics**
- Deterministic: move and merge mechanics  
- Stochastic: random tile insertion (90% = 2, 10% = 4)  
- Terminal state: no valid actions available

---

## 🧮 Learning Algorithm

### Afterstate Value Learning

Instead of learning state values directly, the agent evaluates **afterstates** — board configurations immediately after the agent's action and before random tile placement.

This separation:
- Reduces variance in TD updates
- Decouples agent decisions from environmental randomness
- Improves learning stability in stochastic settings

**TD(0) Update**
```
δ = r + γ · max V(s') − V(s)
```
- Discount factor γ = 1.0 (episodic task)

---

### N-tuple Feature Representation

The value of an afterstate is approximated as a sum of local pattern values:

- Rows (4 tiles each)
- 2×2 squares
- 2×3 rectangles
- L-shaped corner patterns

Each pattern is encoded using base-4 indexing over the log-scaled tiles.

#### Symmetry Exploitation
For each pattern, all **8 symmetric transformations** (rotations and reflections) are used:
- Reduces effective parameter count
- Improves generalization
- Accelerates learning

**Total parameters**: ~15.79 million (lookup tables)

---

### Policy & Exploration

- ε-greedy policy
- Initial ε = 0.5
- Decay: ε × 0.995 per episode
- Minimum ε = 0.001
- Greedy action selected by maximizing afterstate value

---

## 🏗️ Project Structure

```
Reinforcement-learning-2048_agent/
├── 2048_RL.ipynb              # Full implementation and experiments
├── דוח למידת חיזוקים.pdf       # Academic report (Hebrew)
├── models/                    # Trained agents (Git LFS)
│   ├── agent_lr_0.0025.pkl
│   ├── agent_lr_0.005.pkl
│   ├── agent_lr_0.01.pkl
│   └── agent_lr_0.05.pkl
├── README.md
├── requirements.txt
└── .gitattributes
```

---

## 🛠️ Technology Stack

- **Python 3**
- **NumPy**, **Pandas**
- **Matplotlib**, **Seaborn**
- **Numba** (JIT acceleration)
- **Pygame** (interactive visualization)
- **Jupyter Notebook**

---

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- Git with Git LFS support
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Itamar-Melnik/Reinforcement-learning-2048_agent.git
   cd Reinforcement-learning-2048_agent
   ```

2. **Install Git LFS** (if not already installed)
   ```bash
   git lfs install
   ```

3. **Pull LFS files** (trained models)
   ```bash
   git lfs pull
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook 2048_RL.ipynb
   ```

---

## 🚀 Training Setup

- **4 agents**, different learning rates
- Learning rates tested: `0.0025, 0.005, 0.01, 0.05`
- **50,000 episodes per agent**
- ~6 hours training time per agent
- Models stored using Git LFS (~63MB each)

> Training was intentionally limited to 50K episodes; full convergence was not reached.

---

## 📊 Evaluation Protocol

- **1,000 evaluation episodes**
- Greedy policy (ε = 0)
- Metrics:
  - Average score
  - Maximum tile achieved
  - Success rates for ≥256, ≥512, ≥1024, ≥2048, ≥4096, ≥8192

---

## 📈 Results

### Agent Performance Summary

| Learning Rate | Avg Score | ≥2048 | ≥4096 | ≥8192 | Notes |
|--------------|-----------|-------|-------|-------|------|
| 0.0025 | 41,081 | 56.9% | 29.9% | 0.2% | Slow learning, under-trained |
| **0.005** ⭐ | 53,691 | 55.5% | 53.7% | 1.8% | **Most stable and robust** |
| **0.01** | **62,704** | **72.4%** | **68.4%** | **4.0%** | Highest performance, less robust |
| 0.05 | 47,569 | 47.5% | 47.2% | 0.3% | Unstable, overly aggressive |

---

## 🔍 Learning Rate Analysis

- **LR = 0.01** achieves the highest expected return by aggressively reinforcing high-reward trajectories.
  - Excels in common late-game scenarios
  - More sensitive to rare or adversarial board configurations under limited training

- **LR = 0.005** demonstrates slower but more uniform value propagation.
  - More robust across diverse game states
  - Lower variance and fewer extreme failures
  - Best trade-off for reliability under fixed training budget

- **LR = 0.0025** requires substantially longer training to reach competitive performance.

---

## 📉 Convergence Assessment

- TD error did not fully stabilize for high-performing agents.
- Persistent TD activity suggests continued reshaping of the value function as agents increasingly encounter complex late-game states (4096+).
- Extended training (200K+ episodes) or learning rate decay would likely improve stability and performance.

---

## 🔬 Feature Insights

- Only ~12% of parameters were actively updated during training
- 2×2 square patterns contributed most strongly to value estimation
- Larger patterns (2×3) provided less marginal benefit
- Symmetry exploitation significantly improved sample efficiency

---

## ⚠️ Limitations

- No explicit robustness metric (e.g., CVaR or percentile-based scores)
- No learning rate schedules or adaptive optimization
- Training duration limited for computational reasons
- Evaluation focused on maximum tile and mean performance

---

## 🎓 Academic Context

This project was developed as part of academic coursework in Reinforcement Learning and serves as a reference implementation of classical value-based RL methods applied to a challenging stochastic environment.

**Language Note**: Code and notebook are in English; academic report is in Hebrew.

---

## 📦 Models

All trained models are stored using Git LFS:
- Conservative, stable, and aggressive learning profiles included
- Each model contains ~15.79M LUT parameters (~63MB file size)

---

## 📌 Key Takeaways

1. Classical RL methods remain competitive with proper feature engineering
2. Afterstate learning significantly reduces variance in stochastic domains
3. Learning rate strongly affects robustness vs. peak performance
4. Symmetry exploitation is critical for scalability
5. Training time remains the dominant performance bottleneck

---

## 👥 Contributors

Developed by Itamar Melnik as part of academic coursework in Reinforcement Learning.

## 🔗 Repository

[https://github.com/Itamar-Melnik/Reinforcement-learning-2048_agent](https://github.com/Itamar-Melnik/Reinforcement-learning-2048_agent)

---

**Status**: Academic project — archived for reference and educational purposes
