# 🎮 Reinforcement Learning Agent for 2048 Game

Implementation of an advanced Deep Q-Learning agent trained to play the classic 2048 puzzle game. The agent learns optimal strategies through trial-and-error to maximize scores and reach high-value tiles (2048, 4096, and beyond).

## 🧠 What's Inside

- **`2048_RL.ipynb`** - Main Jupyter Notebook with the complete implementation:
  - Custom 2048 Environment: Board logic, move execution, tile merging, and custom reward design
  - Dueling Double DQN (D3QN): Advanced neural network architecture to decouple state-value and advantage functions
  - Prioritized Experience Replay (PER): Efficient learning by focusing on significant game transitions
  - Training & Evaluation: Epsilon-greedy exploration with decay, score tracking, and performance visualization

- **`דוח למידת חיזוקים.pdf`** - Detailed project report in Hebrew covering:
  - Methodology: Technical deep dive into D3QN, state representation, and reward functions
  - Experiments: Hyperparameter tuning (Learning Rates, Batch Sizes)
  - Results: Comparative analysis and conclusions

- **`models/`** - Directory containing pre-trained model checkpoints (`.pkl` files) managed via Git LFS

- **`requirements.txt`** - Required Python packages (numpy, torch/tensorflow, matplotlib, etc.)

## 🎯 Project Goal

Demonstrate core reinforcement learning concepts in a stochastic, discrete environment:

- Implement a robust D3QN architecture from scratch
- Handle a large state space efficiently using neural network approximation
- Optimize hyperparameters to achieve consistent high-tier performance
- Analyze agent improvement over random play and training convergence

## 🛠 Technologies

- Python & Jupyter Notebook
- PyTorch / TensorFlow (Deep Learning framework)
- NumPy (Board & math operations)
- Matplotlib (Visualization of training progress & results)
- Git LFS (Large File Storage for models)

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Itamar-Melnik/Reinforcement-learning-2048_agent.git
   cd Reinforcement-learning-2048_agent
   ```

2. **Pull the models (Git LFS):**
   ```bash
   git lfs pull
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Notebook:**
   Open `2048_RL.ipynb` to train the agent or evaluate the pre-trained models.

## 📈 Key Results

The agent demonstrates significant mastery over the game mechanics:

- **Best Configuration (LR=0.01):** Achieved the 4096 tile in 68.4% of games
- **High-Tier Performance:** Frequent reaching of 2048 and occasional 8192 tiles
- **Convergence:** Clear improvement over random play, showing stable learning and effective "corner strategy" development


## 💡 Key Learnings

- Balancing Exploration vs. Exploitation in high-dimensional state spaces
- The impact of Dueling architectures on reducing Q-value overestimation
- Designing Reward Shaping to encourage tile merging and board organization

