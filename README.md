# Learning Robust Voting Rules with Adversarial Graph Neural Networks

This repository contains the codebase for automated discovery of welfare-maximising voting rules with graph neural networks. Our method enforces
- Voter **anonymity**
- Candidate **neutrality**
- Mechanism **monotonicity**
- **Generalisation** to an arbitrary number of voter and candidates
- **Robustness** to strategic voting

## Setup

We use [uv](https://docs.astral.sh/uv/) for dependency management. After installation, first run `uv sync` to install all required packages. Pytorch scatter will need to then be manually installed, with `uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html`.

## Experiments

To manually download datasets, see the README file in the data folder.

To run experiments, go into the relevant experiments folder, and run `uv run python main.py`. Configurations and sweeps are managed using [hydra](https://hydra.cc/docs/1.3/intro/).