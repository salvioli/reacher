# reacher
A deep reinforcement learning agent implementing the control of a simple two joints robotic arm.

## Project details
This project was completed as part of the [deep reinforcement learning nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

### Task description
In this task a double-jointed arm should move to target locations.
The following is the description of the reward function together with the observation and actions space.
- __reward__ is +0.1 for each step that the agent's hand is in the goal location

- __observation space__ consists of 33 variables corresponding to position, rotation, velocity, angular velocities.

- __action space__ is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic.

### Solution criteria
In order to solve the environment the agent must get an average score of +30 over 100 consecutive episodes.

## Getting started
### Prerequisites
A working python 3 environment is required. You can easily setup one installing [anaconda] (https://www.anaconda.com/download/)

It is suggested to create a new environment as follows:
```bash
conda create --name reacher python=3.6
```
activate the environment
```bash
source activate reacher
```
start the jupyter server
```bash
python jupyter-notebook --no-browser --ip 127.0.0.1 --port 8888 --port-retries=0
```
### Installation and execution
1. Download the pre-compiled unity environment
Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
1. Decompress the archive at your preferred location (e.g. in this repository working copy)
1. Open Report.ipynb notebook 
1. Insert your path to the pre-compiled unity environment to allow the notebook to run it
1. Run the Report.ipynb to install all remaining dependencies and explore my project work