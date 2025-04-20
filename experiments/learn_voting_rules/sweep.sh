uv run python main.py hydra.mode=MULTIRUN \
    'voting_rule=plurality,borda,copeland,minimax,stv' \
    'representation=graph,set' \
    'model_size=small,medium' \
    '+repeat_number=range(10)'