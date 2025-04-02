import os
import warnings
import math

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
from geometric_governance.train import (
    compute_rule_loss,
    compute_welfare_loss,
    compute_monotonicity_loss,
    make_optim_and_scheduler,
)
from geometric_governance.model import ElectionModel, create_election_model
from conf.schema import Config
from dataset import Dataloader, load_dataloader


def run_evaluation(
    dataloader: Dataloader, model: ElectionModel, cfg: Config
) -> tuple[float, float]:
    model.eval()
    mean_welfare = 0.0
    total, correct = 0, 0
    with torch.no_grad():
        for data_ in dataloader:
            data = data_.to(device=device)
            election = model.election(data)
            welfare = (
                torch.exp(election.log_probs) * data.welfare
            ).sum().item() / cfg.dataloader_batch_size
            mean_welfare += welfare

            total += data.winners.sum().item()

            correct += ((election.winners > 0) & (data.winners > 0)).sum().item()

    mean_welfare /= len(dataloader)
    accuracy = correct / total
    return mean_welfare, accuracy


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    cfg = omega_to_pydantic(cfg, Config)

    # Set up datasets
    train_dataloader, val_dataloader, test_dataloader = (
        load_dataloader(
            welfare_rule=cfg.welfare_rule,
            vote_data=cfg.vote_data,
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
        representation="graph",
        num_candidates=get_max(cfg.train_dataset.num_candidates),
        model_size=cfg.model_size,
        aggr=cfg.model_aggr,
    ).to(device=device)
    print(f"parameter_count: {get_parameter_count(model)}")

    optim, scheduler = make_optim_and_scheduler(model, lr=cfg.learning_rate)

    method = "welfare" if cfg.welfare_loss_enable else "rule"
    experiment_name = (
        f"{cfg.vote_data}-{cfg.welfare_rule}-{method}-{cfg.model_size}-{cfg.model_aggr}"
    )
    logger = Logger(
        experiment_name=experiment_name,
        config=cfg,
        mode=cfg.logging_mode,
    )
    logger.begin()
    with tqdm(range(cfg.train_num_epochs)) as pbar:
        best_validation_welfare: float = -math.inf

        for epoch in range(cfg.train_num_epochs):
            # Train
            train_loss = 0
            train_rule_loss = 0
            train_welfare_loss = 0
            train_monotonicity_loss = 0
            train_welfare = 0
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

                # Welfare loss
                welfare_loss = compute_welfare_loss(
                    election=election,
                    welfare=data.welfare,
                    batch_size=cfg.dataloader_batch_size,
                )
                train_welfare_loss += welfare_loss.item()

                # Choose loss
                if cfg.welfare_loss_enable:
                    loss += welfare_loss
                else:
                    loss += rule_loss

                # Monotonicity loss
                if cfg.monotonicity_loss_enable:
                    monotonicity_loss = compute_monotonicity_loss(
                        election, data, batch_size=cfg.monotonicity_loss_batch_size
                    )
                    train_monotonicity_loss += monotonicity_loss.item()
                    loss += monotonicity_loss
                    train_monotonicity_loss += monotonicity_loss.item()

                # Total Loss
                train_loss += loss.item()

                # Update weights
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                optim.step()
                scheduler.step()

                # Logging
                # Accuracy
                total += data.winners.sum().item()
                correct += ((election.winners > 0) & (data.winners > 0)).sum().item()
                # Welfare
                with torch.no_grad():
                    welfare = (
                        data.welfare * torch.exp(election.log_probs)
                    ).sum() / cfg.dataloader_batch_size
                    train_welfare += welfare.item()

            train_loss /= cfg.train_iterations_per_epoch
            train_rule_loss /= cfg.train_iterations_per_epoch
            train_welfare_loss /= cfg.train_iterations_per_epoch
            train_welfare /= cfg.train_iterations_per_epoch
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
                    "train/welfare_loss": train_welfare_loss,
                    "train/monotonicity_loss": train_monotonicity_loss,
                    "train/accuracy": train_accuracy,
                }
            )

            # Validation
            model.eval()
            val_welfare, val_accuracy = run_evaluation(val_dataloader, model, cfg)

            if val_welfare > best_validation_welfare:
                print(f"New best welfare: {val_welfare}, accuracy {val_accuracy}")
                logger.summary["best_validation_epoch"] = epoch
                logger.summary["best_validation_accuracy"] = val_accuracy
                logger.summary["best_validation_welfare"] = val_welfare
                torch.save(
                    model,
                    os.path.join(logger.checkpoint_dir, "model_best.pt"),
                )
                best_validation_welfare = val_welfare

            logger.log(
                {
                    "val/welfare": val_welfare,
                    "val/accuracy": val_accuracy,
                }
            )
            logger.commit()

            pbar.set_postfix(
                {
                    "train_welfare": train_welfare,
                    "train_accuracy": train_accuracy,
                    "val_welfare": val_welfare,
                    "val_accuracy": val_accuracy,
                }
            )

            pbar.update(1)

    # Candidate number generalisation test
    model = torch.load(
        os.path.join(logger.checkpoint_dir, "model_best.pt"), weights_only=False
    )
    test_welfare, test_accuracy = run_evaluation(test_dataloader, model, cfg)
    logger.summary["test_loss"] = test_welfare
    logger.summary["test_accuracy"] = test_accuracy
    print(f"Test welfare: {test_welfare} Test accuracy: {test_accuracy}")
    logger.commit()
    logger.close()


if __name__ == "__main__":
    main()
