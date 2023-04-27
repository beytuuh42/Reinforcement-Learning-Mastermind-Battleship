# Reinforcement Learning with strategic board games

## Contents

### Agents

This folder contains a jupyter file that uses `[DQN, DDQN, PPO & Reinforce]` implementation from the [tf_agents](https://github.com/tensorflow/agents) library to play the games.

### Environments

This folder contains the implementation of the games Battleship and Mastermind. They implement PyEnvironment, so they can be used with the `tf_agents` library.

## Requirements

This project only works on a Linux OS, because the agents use [deepmind's reverb](https://github.com/deepmind/reverb) for data collection.

## Installation

### Docker

The recommended way is to use docker.

Clone the repository and run the batch file "start.bat", which will build an image from the Dockerfile and run a container. If you are not on Windows, than copy the commands from the file and paste them in your native terminal.
When running the script, a link to the deployed notebook will appear in the terminal. Simply copy & paste that into the browser and run the .ipynb file, which is located in the `agents` folder.

### Linux

- Works with Python 3.8.10
- Add the additional argument `mask=None` to the `call` function in `tf_agents/networks/value_network.py`, see [here](https://github.com/tensorflow/agents/issues/762)
  - Or run this command `sed -i 's/training=False/training=False, mask=None/g'` _PATH_TO_PYTHON_PACKAGES_`/tf_agents/networks/value_network.py`

### Windows

See [Docker](#docker).