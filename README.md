
# Frozen Lake Q-Learning Agent
![Alt Text](https://im7.ezgif.com/tmp/ezgif-7-825f30ae62.gif)
## Overview
This repository contains a Python implementation of a Q-Learning agent for the "FrozenLake-v1" environment provided by [Gymnasium](https://www.gymnasium.openai.com/). The agent applies reinforcement learning to learn the best strategy for crossing a grid while avoiding holes. This implementation features functions for epsilon-greedy policy selection, Q-Learning, policy evaluation, and a demonstration of the agent's performance.

## Features
- **Epsilon-Greedy Policy**: Balances between exploration and exploitation, selecting actions based on the epsilon-greedy approach.
- **Q-Learning**: An off-policy algorithm to determine the optimal action-value function.
- **Policy Evaluation**: Assesses the learned policy's effectiveness.
- **Agent Demonstration**: Visually showcases the agent's ability in the environment.

## Installation

### Prerequisites
- Python 3.x
- pip (Python package installer)

### Steps
1. **Install Python**: Ensure that Python 3.x is installed on your system. You can download it from [python.org](https://www.python.org/).

2. **Install Dependencies**: This project depends on several Python libraries, including Gymnasium (formerly known as "Gym"). Install them using pip:

    ```bash
    pip install numpy gymnasium tqdm
    ```

    This command installs `numpy` for numerical operations, `gymnasium` for the environment, and `tqdm` for the progress bar functionality.

3. **Clone the Repository**: Clone this repository to your local machine using:

    ```bash
    git clone <repository-url>
    ```
    Replace `<repository-url>` with the URL of this GitHub repository.

## Usage
Run the `FINAL_Frozen_Lake.py` script to train and evaluate the Q-Learning agent in the Frozen Lake environment:

```bash
python FINAL_Frozen_Lake.py
```

This will initiate the training process for the agent, evaluate its performance, and run a demonstration in the environment.

## Files in the Repository
- `FINAL_Frozen_Lake.py`: Contains all functions and the main routine for executing the Q-Learning algorithm.
- `README.md`: Provides project details and usage instructions.

