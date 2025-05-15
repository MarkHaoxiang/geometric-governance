# Learning Robust Voting Rules with Adversarial Graph Neural Networks

This repository contains the codebase for automated discovery of welfare-maximising voting rules with graph neural networks. Our method enforces
- Voter **anonymity**
- Candidate **neutrality**
- Other **voting criteria** (e.g. monotonicity)
- **Generalisation** to an arbitrary number of voter and candidates
- **Robustness** to strategic voting

The diagrams below showcase an overview of our system consisting of the Graph Election Strategy Network (GESN) and Graph Election Voting Network (GEVN) architectures (Left), as well as our Election Bipartite Graph (EBG) election representation (Right).

<p align="center">
    <img src="./figures/overview.svg" width="50%"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="./figures/bipartite_preference_profile.svg" width="20%">
</p>

*In the face of adverse motives, it is indispensable to achieve a consensus. Elections have been the canonical way by which modern democracy has operated since the 17th century. Nowadays, they regulate markets, provide an engine for modern recommender systems or peer-to-peer networks, and remain the main approach to represent democracy. However, a desirable universal voting rule that satisfies all hypothetical scenarios is still a challenging topic, and the design of these systems is at the forefront of mechanism design research. Automated mechanism design is a promising approach, and recent works have demonstrated that set-invariant architectures are uniquely suited to modelling electoral systems. However, various concerns prevent the direct application to real-world settings---in particular, robustness to strategic voting. In this paper, *we generalise the expressive capability of learned voting rules, and combine improvements in neural network architecture with adversarial training to improve the robustness of voting rules while maximizing social welfare*. We evaluate the effectiveness of our methods on both synthetic and real-world datasets. _Our method resolves critical limitations of prior work regarding learning voting rules by representing elections using bipartite graphs, and learning such voting rules using graph neural networks. We believe this opens new frontiers for applying machine learning to real-world elections_*

## Setup

We use [uv](https://docs.astral.sh/uv/) for dependency management. After installation, first run `uv sync` to install all required packages. 

Enter the virtual environment by running `. .venv/bin/activate`.

Pytorch scatter will need to then be manually installed, with `uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html`.

## Experiments

To manually download datasets, see the README file in the data folder.

To run experiments, go into the relevant experiments folder, and run `uv run python main.py`. Configurations and sweeps are managed using [hydra](https://hydra.cc/docs/1.3/intro/).

Preconfigured sweeps can be run by going into the relevant experiments folder, and running `./sweeps/SWEEP_NAME.sh`.

Experiments include **learning voting rules** (experiments/learn_voting_rules), **maximising welfare** (experiments/maximise_welfare), and **strategic voting** (experiments/strategic_voting).
