from collections import defaultdict
import numpy as np
import wandb

project_name = "geometric-governance/learn_voting_rules"
api = wandb.Api()


def fmt(accs: list[float]) -> str:
    if not accs:
        return "[-]"
    n = len(accs)
    mean = np.mean(accs)
    if n == 1:
        return f"${mean:.2f}$"  # No CI possible
    se = np.std(accs, ddof=1) / np.sqrt(n)
    ci95 = 1.96 * se
    return f"${mean:.2f} pm {ci95:.2f}$"


def cell(rule, repr_, size, split):
    return fmt(results[rule][repr_][size][split])


if __name__ == "__main__":
    # Structure: results[voting_rule][representation][model_size][split] = list of accuracies
    results = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )  # type: ignore

    for run in api.runs(project_name):
        if run.state != "finished":
            continue

        cfg = run.config
        summary = run.summary

        rule = cfg.get("voting_rule")
        repr_ = cfg.get("representation")
        size = cfg.get("model_size")
        val_acc = summary.get("best_validation_accuracy")
        test_acc = summary.get("test_accuracy")

        if rule and repr_ and size and val_acc is not None:
            results[rule][repr_][size]["val"].append(val_acc)
            if test_acc is not None:
                results[rule][repr_][size]["test"].append(test_acc)

    rules = ["plurality", "borda", "copeland", "minimax", "stv"]
    cols = [
        ("graph", "medium"),  # GEVN
        ("graph", "small"),  # GEVN (Small)
        # ("graph_unnormalised", "medium"),
        # ("graph_unnormalised", "small"),
        ("set", "medium"),  # DeepSets
        ("set", "small"),  # DeepSets (Small)
        ("set_one_hot", "medium"),
        ("set_one_hot", "small"),
    ]

    table_str = f"""table(
        columns: 8,
        table.header(
            table.vline(stroke: 0.5pt), [],[], table.vline(stroke: 0.5pt),
            [GEVN], [GEVN (Small)],
            [DeepSets], [DeepSets (Small)],
            [DeepSets OneHot], [DeepSets OneHot (Small)],
            table.vline(stroke: 0.5pt)
        ),
    {
        "".join(
            f'''
        [#rule("{rule}")],
        [Validation], {", ".join(f"[{cell(rule, repr_, size, 'val')}]" for repr_, size in cols)},
        [Test],       {", ".join(f"[{cell(rule, repr_, size, 'test')}]" for repr_, size in cols)},
        table.hline(stroke: 0.5pt),'''
            for rule in rules
        )
    }
    )
    """

    print(table_str)
