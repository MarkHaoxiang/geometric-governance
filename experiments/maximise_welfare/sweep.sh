uv run python main.py hydra.mode=MULTIRUN \
    'welfare_rule=utilitarian,nash,rawlsian' \
    'welfare_loss_enable=true,false' \
    'vote_data=ranking,utility' \
    '+repeat_number=range(5)'