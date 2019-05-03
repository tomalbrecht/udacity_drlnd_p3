# Project 3: Collaboration and Competition

## Project Details

This project is based on my former project [`Continuous Control`](https://github.com/tomalbrecht/udacity_drlnd_p2) from this drlnd course. I used this as base project, because I already implemented the code to run multiple agents.

I chose the DDPG (Deep Deterministic Policy Gradients) algorithm because it is able to handle continuous spaces, which is needed for this environment and seemed easier as discretization (see Chapter 1 of the course). Continuous spaces make it more difficult to train an agent, because the action space gets highly dimensional. In contrast DQN (with Q-tables)solves problems with high-dimensional observation spaces, but it can only handle discrete and low-dimensional action spaces. Using a neural network to approximate these values in a convinient way.

The algorithm also benefits from two separate neural network (actor and critic) - so the target network will only be updated with every second training step (see hyperparameters for details). 

## State and Action Space

This project uses an adapted [`tennis environment`](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) from unity.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

* Set-up: Two-player game where agents control rackets to bounce ball over a net.
* Goal: The agents must bounce ball between one another while not dropping or sending ball out of bounds.
* Agents: The environment contains two agent linked to a single Brain named TennisBrain. After training you can attach another Brain named MyBrain to one of the agent to play against your trained model.
* Agent Reward Function (independent):
    * +0.1 To agent when hitting ball over net.
    * -0.1 To agent who let ball hit their ground, or hit ball out of bounds.
* Brains: One Brain with the following observation/action space.
* Vector Observation space: 8 variables corresponding to position and velocity of ball and racket.
* Vector Action space: (Continuous) Size of 2, corresponding to movement toward net or away from net, and jumping.
* Visual Observations: None.
* Reset Parameters: One, corresponding to size of ball.
* Benchmark Mean Reward: 2.5

## Getting Started

### Step 1: Activate the Environment
If you haven't already, please follow the instructions in the [`DRLND GitHub repository`](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. These instructions can be found in README.md at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

(For Windows users) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

### Step 2: Download the Unity Environment
For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

Linux: [`click here`](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
Mac OSX: [`click here`](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
Windows (32-bit): [`click here`](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
Windows (64-bit): [`click here`](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
Then, place the file in the p3_tennis/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

(For Windows users) Check out [`this link`](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not [`enabled a virtual screen`](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [`this link`](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to [`enable a virtual screen`](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the Linux operating system above.)

### Instructions

After the setup open and run `Tennis.ipynb` notebook using the drlnd kernel to train the DDPG agent. Follow the instructions there.

Once trained the model weights will be saved in the same directory in the files `checkpoint_actor.pth` and `checkpoint_critic.pth`.

The model weights are used by the `Tennis_run.ipynb` notebook in the simulator.