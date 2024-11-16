# Continuous Control with DDPG

## Project Overview

This project implements a Deep Deterministic Policy Gradient (DDPG) agent to solve the Reacher environment, where a double-jointed arm needs to maintain position at target locations. The agent learns to control the robotic arm in a continuous action space to maximize rewards.

**Environment Details:**
- **State Space**: 33 variables (position, rotation, velocity, angular velocities)
- **Action Space**: 4 continuous values between -1 and 1 (torque for joints)
- **Reward**: +0.1 for each step the agent's hand is in the goal location
- **Goal**: Achieve average score of +30 over 100 consecutive episodes

<p align="center">
  <img src="https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif" alt="Trained Agent" width="500"/>
</p>

## Dependencies and Installation

### 1. Python Environment
```bash
# Create and activate a new environment with Python 3.6
conda create --name continuous-control python=3.6
conda activate continuous-control
```

### 2. Required Packages
```bash
# PyTorch (with CUDA support for GPU)
conda install pytorch cudatoolkit=10.1 -c pytorch

# Additional dependencies
pip install numpy matplotlib jupyter unityagents
```

### 3. Unity Environment
Download the environment that matches your operating system:

**Version 1: Single Agent**
- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

**Version 2: Twenty Agents**
- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

After downloading, place the file in the project root directory and unzip it.

## Running the Project

1. **Clone the Repository**
```bash
git clone git@github.com:Chaklader/DPRL-Policy-based-methods.git
cd DPRL-Policy-based-methods
```

2. **Start Jupyter Notebook**
```bash
jupyter notebook
```

3. **Open and Run the Notebook**
- Open `Continuous_Control.ipynb`
- Select `Kernel > Restart & Run All` to run all cells
- The training will begin and run until the environment is solved (average score of 30 over 100 episodes)
- Trained model weights will be saved as `actor_checkpoint.pth` and `critic_checkpoint.pth`

**Note**: Training typically takes around 170-200 episodes to solve the environment. The notebook includes visualization of the training progress and final performance metrics.