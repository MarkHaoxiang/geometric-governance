#!/bin/bash

WELFARE_FNS=(utilitarian nash rawlsian)

for fn in "${WELFARE_FNS[@]}"; do
  echo "Launching job for welfare fn=$fn"
  uv run python main.py -m -cn synthetic \
    "vote_source=dirichlet,spatial" \
    "election_model.freeze_weights=false,true" \
    "welfare_rule=$fn" \
    "project_name=results" \
    "strategy_voter_information=results" \
    "+repeat_number=range(5)" &
done

wait