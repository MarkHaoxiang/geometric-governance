import os
import warnings

import hydra
import torch
import torch.optim as o
from tqdm import tqdm


from geometric_governance.util import (
    Logger,
    get_max,
    get_parameter_count,
    omega_to_pydantic,
    cuda as device,
)
from geometric_governance.train import compute_rule_loss, compute_monotonicity_loss
from geometric_governance.model import ElectionModel, create_election_model
from conf.schema import Config
from dataset import Dataloader, load_dataloader


def run_evaluation(
    dataloader: Dataloader, model: ElectionModel, cfg: Config
) -> tuple[float, float]:
    model.eval()
    loss = 0.0
    total, correct = 0, 0
    with torch.no_grad():
        for data_ in dataloader:
            data = data_.to(device=device)
            election = model.election(data)
            rule_loss = compute_rule_loss(
                election, data.winners, cfg.dataloader_batch_size
            )
            total += data.winners.sum().item()
            correct += ((election.winners > 0) & (data.winners > 0)).sum().item()
            loss += rule_loss.item()

    loss /= len(dataloader)
    accuracy = correct / total
    return loss, accuracy


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    cfg = omega_to_pydantic(cfg, Config)

    # Override because DeepSets can't generalise
    if cfg.representation == "set":
        cfg.val_dataset.num_candidates = get_max(cfg.train_dataset.num_candidates)

    # Set up datasets
    train_dataloader, val_dataloader, test_dataloader = (
        load_dataloader(
            voting_rule=cfg.voting_rule,
            representation=cfg.representation,
            dataloader_batch_size=cfg.dataloader_batch_size,
            **dataset.model_dump(),
        )
        for dataset in (cfg.train_dataset, cfg.val_dataset, cfg.test_dataset)
    )

    if cfg.train_iterations_per_epoch > len(train_dataloader):
        cfg.train_iterations_per_epoch = len(train_dataloader)
        warnings.warn(
            f"train_iterations_per_epoch override to {cfg.train_iterations_per_epoch} as it is larger than the dataset size"
        )

    # Create model
    model = create_election_model(
        representation=cfg.representation,
        num_candidates=get_max(cfg.train_dataset.num_candidates),
    ).to(device=device)
    print(f"parameter_count: {get_parameter_count(model)}")

    optim = o.Adam(model.parameters(), lr=cfg.learning_rate)

    warmup_epochs = 5
    warmup_scheduler = o.lr_scheduler.LinearLR(
        optim, start_factor=0.1, end_factor=1, total_iters=warmup_epochs
    )
    main_scheduler = o.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=5, T_mult=2)
    scheduler = o.lr_scheduler.SequentialLR(
        optim,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )

    experiment_name = f"{cfg.representation}-election-{cfg.voting_rule}"
    logger = Logger(
        experiment_name=experiment_name,
        config=cfg,
        mode=cfg.logging_mode,
    )
    logger.begin()
    with tqdm(range(cfg.train_num_epochs)) as pbar:
        best_validation_accuracy: float = 0.0

        for epoch in range(cfg.train_num_epochs):
            # Train
            train_loss = 0
            train_rule_loss = 0
            train_monotonicity_loss = 0
            total, correct = 0, 0

            model.train()

            train_iter = iter(train_dataloader)

            for _ in range(cfg.train_iterations_per_epoch):
                optim.zero_grad()
                loss = 0
                data = next(train_iter).to(device)
                election = model.election(data)

                # Rule Loss (NLLL)
                rule_loss = compute_rule_loss(
                    election=election,
                    winners=data.winners,
                    batch_size=cfg.dataloader_batch_size,
                )
                train_rule_loss += rule_loss.item()
                loss += rule_loss

                # Monotonicity loss
                if cfg.monotonicity_loss_enable:
                    monotonicity_loss = compute_monotonicity_loss(
                        election, data, batch_size=cfg.monotonicity_loss_batch_size
                    )
                    train_monotonicity_loss += monotonicity_loss.item()
                    loss += monotonicity_loss

                # Update weights
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                optim.step()
                scheduler.step()

                total += data.winners.sum().item()
                correct += ((election.winners > 0) & (data.winners > 0)).sum().item()

                train_loss += loss.item()
                if cfg.monotonicity_loss_enable:
                    train_monotonicity_loss += monotonicity_loss.item()

            train_loss /= cfg.train_iterations_per_epoch
            train_rule_loss /= cfg.train_iterations_per_epoch
            train_monotonicity_loss /= cfg.train_iterations_per_epoch
            train_accuracy = correct / total

            if epoch % cfg.logging_checkpoint_interval == 0:
                torch.save(
                    model,
                    os.path.join(logger.checkpoint_dir, f"model_{epoch}.pt"),
                )

            logger.log(
                {
                    "train/total_loss": train_loss,
                    "train/rule_loss": train_rule_loss,
                    "train/monotonicity_loss": train_monotonicity_loss,
                    "train/accuracy": train_accuracy,
                }
            )

            # Validation
            model.eval()
            val_loss, val_accuracy = run_evaluation(val_dataloader, model, cfg)

            if val_accuracy > best_validation_accuracy:
                print(f"New best accuracy: {val_accuracy}")
                logger.summary["best_validation_epoch"] = epoch
                logger.summary["best_validation_accuracy"] = val_accuracy
                logger.summary["best_validation_loss"] = val_loss
                torch.save(
                    model,
                    os.path.join(logger.checkpoint_dir, "model_best.pt"),
                )
                best_validation_accuracy = val_accuracy

            logger.log(
                {
                    "val/rule_loss": val_loss,
                    "val/accuracy": val_accuracy,
                }
            )
            logger.commit()

            pbar.set_postfix(
                {
                    "train_rule_loss": train_rule_loss,
                    "train_accuracy": train_accuracy,
                    "val_rule_loss": val_loss,
                    "val_accuracy": val_accuracy,
                }
            )

            pbar.update(1)

    # Candidate number generalisation test
    if cfg.representation == "graph":
        model = torch.load(
            os.path.join(logger.checkpoint_dir, "model_best.pt"), weights_only=False
        )
        test_loss, test_accuracy = run_evaluation(test_dataloader, model, cfg)
        logger.summary["test_loss"] = test_loss
        logger.summary["test_accuracy"] = test_accuracy
        print(f"Test loss: {test_loss} Test accuracy: {test_accuracy}")
    logger.commit()
    logger.close()


if __name__ == "__main__":
    main()
