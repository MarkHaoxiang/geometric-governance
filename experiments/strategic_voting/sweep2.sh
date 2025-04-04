uv run python main.py --multirun \
    'welfare_rule=utilitarian' \
    'strategy_module_enable=true' \
    'election_model.from_pretrained=utilitarian_small' \
    'election_model.freeze_weights=false' \
    '+repeat_number=range(5)'