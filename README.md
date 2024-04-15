# PPO
An implementation of Proximal Policy Optimization (PPO) using Generalized Advantage Estimation (GAE) and multi-processing.

## Installing

`python3 -m venv venv` to set up a virtual environment 

`cd pip install .` to install, or `pip install -e .` for development.

`python src/run_ppo.py config/pendulum.yaml` trains PPO for a given config file. Examples for different environments with hyperparameters I've found that work well can be found in `config/`.

## Paper Notes
See https://salmanmohammadi.github.io/content/ppo/ for an explanation of the method.
