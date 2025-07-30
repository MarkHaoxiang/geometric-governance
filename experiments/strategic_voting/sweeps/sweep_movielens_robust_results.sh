#!/bin/bash

WELFARE_FNS=(utilitarian nash rawlsian)

for fn in "${WELFARE_FNS[@]}"; do
  echo "Launching job for welfare fn=$fn"
  uv run python main.py -m -cn movielens \
    "election_model.freeze_weights=true" \
    "election_model.from_pretrained=robust" \
    "welfare_rule=$fn" \
    "project_name=results" \
    "strategy_voter_information=results" \
    "+repeat_number=range(5)" &
done

wait