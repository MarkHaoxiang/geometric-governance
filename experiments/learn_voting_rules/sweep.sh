uv run python main.py hydra.mode=MULTIRUN \
    'voting_rule=plurality,borda,copeland,minimax,stv' \
    'representation=graph,set' \
    '+repeat_number=range(5)'