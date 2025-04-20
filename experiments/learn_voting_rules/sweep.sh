#!/bin/bash

VOTING_RULES=(plurality borda copeland minimax stv)

for vr in "${VOTING_RULES[@]}"; do
  echo "Launching job for voting_rule=$vr"
  uv run python main.py hydra.mode=MULTIRUN \
    'voting_rule=$vr' \
    'representation=graph,set' \
    'model_size=small,medium' \
    '+repeat_number=range(10)' &
done

wait