#!/bin/bash

WELFARE_FNS=(utilitarian nash rawlsian)

for fn in "${WELFARE_FNS[@]}"; do
  echo "Launching job for welfare fn=$fn"
  uv run python main.py -m -cn base \
    "+vote_data=ranking,utility" \
    "+welfare_loss_enable=true,false" \
    "+welfare_rule=$fn" \
    "+vote_source=movielens" \
    "train_dataset.num_voters=[10,25]" \
    "train_dataset.num_candidates=[3,7]" \
    "train_dataset.dataset_size=5000" \
    "val_dataset.num_voters=30" \
    "val_dataset.num_candidates=10" \
    "val_dataset.dataset_size=512" \
    "test_dataset.num_voters=45" \
    "test_dataset.num_candidates=15" \
    "test_dataset.dataset_size=512" \
    "+repeat_number=range(5)" &
done

wait
