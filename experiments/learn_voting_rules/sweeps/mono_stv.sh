#!/bin/bash
uv run python main.py -m -cn mono \
    voting_rule=stv \
    monotonicity_loss_train=false \
    '+repeat_number=range(5)' &

uv run python main.py -m -cn mono \
    voting_rule=stv \
    monotonicity_loss_train=true \
    '+repeat_number=range(5)' &
  
