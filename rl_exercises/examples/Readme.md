# Useful Examples
This directory contains a small collection of examples around the lecture. You can run a demo of the StableBaselines library, see how the RLiable plotting package works and how to tune your hyperparameters.

## RL Training Demo
Here we show you how to train an agent on an environment using one of the most popular RL libraries.
We train and optimize a BipedalWalker (gymnasium) using a Soft Actor-Critic (SAC, https://arxiv.org/abs/1801.01290).
For this we provide a notebook and a training script you are welcome to checkout.
Training time vary a bit depending on your hardware and settings and can take a while!

### Train
If you want to use the traininc script, activate your conda env and run the following in this dir:
```bash
python train.py
```
Check out the folder `configs` for possible parameters. You can read [here](https://hydra.cc/docs/advanced/override_grammar/basic/) how to set parameters via the commandline.

### Visualize Training Progress
This command will let you see progress in training:
```bash
tensorboard --logdir .
```
Once training is finished, you can check the notebook for a replay.

## RLiable Example
RLiable is a plotting library that handles bootstrapping in plots in an appropriate manner. We have two examples for how to use it, a minimal one and one using the actual agent loop.

### Minimal Example
We use `rliable` to evaluate RL algorithms more robustly. You can find an example script `rliable_example.py` in this folder alongside some dummy data (`demo_data_seed_*.csv`). Run it to see how `rliable` aggregates performance and computes confidence intervals across multiple random seeds!
```bash
python rliable_example.py
```
This example obviously uses dummy data, but should show the general workflow for the package.

### RLiable in the Agent Loop
The agent training script produces two files for reward logging: train_rewards.csv and eval_rewards.csv.
Both are structured similarly to the demo data, so we can treat them in the same way.
For this example, let's create the training data ourselves in a sweep. 
This command will run multiple seeds in sequence and thereby give us the results of multiple runs.
Run this from the top level of the repository:
```bash
python rl_exercises/train_agent.py agent=random seed=0,1,2,3,4 -m
```
The -m flag leads to one run per seed being triggered. 
Each run will have its own result directory from which we can read the files.
Still from the repo top level run:
```bash
python rl_exercises/examples/rliable_agent_loop_example.py"
```
Now you can see both curves plotted in the repository. You can re-use this exact script for any agent you implement in the future.
In that case the plots should also look less chaotic since you won't be random sampling!

## Hyperparameter Tuning Example
For hyperparamter tuning, we recommend using the Hypersweeper interface since it works seamlessly with Hydra. 
We have examples using random search and a Bayesian Optimization algorithm, SMAC.
If possible on your system, you should use SMAC since it yields better results. 

The configuration of the optimization happens fully in Hydra config files. 
As preparation, you should install the hypersweeper package like this:
```bash
uv pip install hypersweeper
```

Then take a look at ../configs/rs.yaml or ../configs/smac.yaml - both will look very similar.
Here you can edit how many runs you want to spend, the default is a very low 5.
It is also important to define what to tune. Since right now there is only a random agent, we tune the random seed (something you should never do again ;D).
This is defined in ../configs/search_space. If you're happy with your options, run this command from the repo top level:
```bash
python rl_exercises/train_agent.py --config-name=smac agent=random search_space=random_seed -m
```
Or alternatively:
```bash
python rl_exercises/train_agent.py --config-name=rs agent=random search_space=random_seed -m
```

You will see a "hpo_results" directory with the runs's results (check the "seed_0" subdirectory in case you don't see the aggregation immediately). From these you can pick the best performance.
Note: the hypersweeper will probably generate many warnings about packages you haven't installed. That doesn't matter, you can ignore these.