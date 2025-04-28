#!/bin/bash

WELFARE_FNS=(utilitarian nash rawlsian)

for fn in "${WELFARE_FNS[@]}"; do
  echo "Launching job for welfare fn=$fn"
  uv run python main.py -m -cn base \
    "+vote_data=ranking" \
    "+welfare_loss_enable=true,false" \
    "+welfare_rule=$fn" \
    "+vote_source=dirichlet,spatial" \
    "+repeat_number=range(10)" &
done

wait