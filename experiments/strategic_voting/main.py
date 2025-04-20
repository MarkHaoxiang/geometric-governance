import os
import warnings
import math

import hydra
import torch
import torch.nn.modules
from torch_geometric.data import Data
from torch_scatter import scatter_log_softmax, scatter_add, scatter_max
from tqdm import tqdm


from geometric_governance.util import (
    Logger,
    get_parameter_count,
    omega_to_pydantic,
    cuda as device,
)
from geometric_governance.train import (
    compute_welfare_loss,
    compute_monotonicity_loss,
    make_optim_and_scheduler,
)
from geometric_governance.model import (
    ElectionModel,
    ElectionResult,
    DeepSetStrategyModel,
    create_election_model,
)
from conf.schema import Config
from dataset import Dataloader, load_dataloader


def run_evaluation(
    dataloader: Dataloader, election_model: ElectionModel, strategy_model, cfg: Config
) -> float:
    election_model.eval()
    strategy_model.eval()
    mean_welfare = 0.0
    with torch.no_grad():
        for data_ in dataloader:
            data = data_.to(device=device)

            # Run strategy model
            voters = (~data.candidate_idxs).nonzero()
            voter_to_batch = data.batch[voters]

            truthful_votes = data.edge_attr
            strategic_votes = strategy_model(data.edge_attr, data.edge_index)

            strategic_voters = (
                torch.rand_like(voters, dtype=torch.float) < cfg.strategy_p
            )

            rand_indices = torch.stack(
                [
                    torch.where(voter_to_batch == batch)[0][
                        torch.randint(0, (voter_to_batch == batch).sum(), (1,))
                    ].squeeze()
                    for batch in range(data.batch[-1] + 1)
                ]
            )
            sampled_voters = voters[rand_indices]
            strategic_voters[rand_indices] = True

            strategy_train_mask = torch.isin(data.edge_index[0], sampled_voters)
            strategy_train_mask = strategy_train_mask.unsqueeze(-1)
            strategic_votes_mask = torch.isin(
                data.edge_index[0], strategic_voters.nonzero()
            ).unsqueeze(-1)

            resulting_votes = torch.where(
                strategic_votes_mask, strategic_votes, truthful_votes
            )
            data.edge_attr = resulting_votes

            # Run election model
            election = election_model.election(data)
            welfare = (
                torch.exp(election.log_probs) * data.welfare
            ).sum().item() / cfg.dataloader_batch_size
            mean_welfare += welfare

    mean_welfare /= len(dataloader)
    return mean_welfare


class ManualElectionModel(ElectionModel):
    def forward(self, data: Data):
        vote_sum = scatter_add(data.edge_attr, index=data.edge_index[0], dim=0)
        vote_sum = vote_sum[data.edge_index[0]]
        normalised_votes = data.edge_attr / torch.maximum(
            vote_sum, torch.ones_like(vote_sum)
        )

        logits = scatter_add(
            src=normalised_votes, index=data.edge_index[1].unsqueeze(-1), dim=0
        )[data.candidate_idxs.nonzero()].squeeze()

        out = scatter_log_softmax(logits, data.batch[data.candidate_idxs])
        return out

    def election(self, data: Data):
        log_probs = self(data)
        winner_idxs = scatter_max(log_probs, data.batch[data.candidate_idxs])[1]
        winners = torch.zeros_like(log_probs)
        winners[winner_idxs] = 1
        return ElectionResult(log_probs, winners)


class NoStrategy(torch.nn.Module):
    def forward(self, edge_attr, edge_index):
        return edge_attr


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    cfg = omega_to_pydantic(cfg, Config)

    # Set up datasets
    train_dataloader, val_dataloader, test_dataloader = (
        load_dataloader(
            welfare_rule=cfg.welfare_rule,
            dataloader_batch_size=cfg.dataloader_batch_size,
            **dataset.model_dump(),
        )
        for dataset in (cfg.train_dataset, cfg.val_dataset, cfg.test_dataset)
    )

    if cfg.train.iterations_per_epoch > len(train_dataloader):
        cfg.train.iterations_per_epoch = len(train_dataloader)
        warnings.warn(
            f"train_iterations_per_epoch override to {cfg.train.iterations_per_epoch} as it is larger than the dataset size"
        )

    # Election model definition
    if cfg.election_model.use_manual_election:
        election_model = ManualElectionModel()
        model_name = "manual"
    else:
        if cfg.election_model.from_pretrained:
            model_name = f"{cfg.election_model.from_pretrained}.pt"
            model_path = os.path.join("pretrained_models", model_name)
            if not os.path.exists(model_path):
                raise ValueError(f"{model_name} not found.")
            election_model = torch.load(model_path, weights_only=False)
            election_model.to(device=device)
        else:
            election_model = create_election_model(
                representation="graph", model_size=cfg.election_model.size
            ).to(device=device)
            model_name = f"gevn-{cfg.election_model.size}"
        print(f"election parameter_count: {get_parameter_count(election_model)}")

    # Strategy model definition
    if cfg.strategy_module_enable:
        strategy_model = DeepSetStrategyModel(edge_dim=1, emb_dim=32).to(device=device)
        print(f"strategy parameter_count: {get_parameter_count(strategy_model)}")
    else:
        strategy_model = NoStrategy()

    # Optimiser and Scheduler
    is_training_election = (
        not cfg.election_model.freeze_weights
        and not cfg.election_model.use_manual_election
    )
    if is_training_election:
        e_optim, e_scheduler = make_optim_and_scheduler(
            election_model,
            lr=cfg.train.learning_rate,
            total_epochs=cfg.train.num_epochs,
            warmup_epochs=cfg.train.learning_rate_warmup_epochs,
            warm_restart=cfg.train.learning_rate_warm_restart,
        )
    if cfg.strategy_module_enable:
        s_optim, s_scheduler = make_optim_and_scheduler(
            strategy_model,
            lr=cfg.train.learning_rate,
            total_epochs=cfg.train.num_epochs,
            warmup_epochs=cfg.train.learning_rate_warmup_epochs,
            warm_restart=cfg.train.learning_rate_warm_restart,
        )

    freeze = "freeze" if cfg.election_model.freeze_weights else "train"
    experiment_name = (
        f"strategy-{cfg.welfare_rule}-{cfg.election_model.size}-{freeze}-{model_name}"
    )
    logger = Logger(
        experiment_name=experiment_name,
        config=cfg,
        mode=cfg.logging_mode,
    )
    logger.begin()
    with tqdm(range(cfg.train.num_epochs)) as pbar:
        election_model.train()
        strategy_model.train()
        for epoch in range(cfg.train.num_epochs):
            # Train
            train_loss = 0
            train_welfare_loss = 0
            train_strategy_loss = 0
            train_monotonicity_loss = 0
            train_welfare = 0

            election_model.train()

            train_iter = iter(train_dataloader)

            for _ in range(cfg.train.iterations_per_epoch):
                election_loss = 0
                data = next(train_iter).to(device)

                # Strategy Loss
                strategy_loss = 0
                if cfg.strategy_module_enable:
                    voters = (~data.candidate_idxs).nonzero()
                    candidate_idxs_nonzero = data.candidate_idxs.nonzero()
                    voter_to_batch = data.batch[voters]

                    truthful_votes = data.edge_attr
                    strategic_votes = strategy_model(data.edge_attr, data.edge_index)
                    strategic_votes_detached = strategic_votes.detach()

                    # p% of voters are strategic
                    strategic_voters = (
                        torch.rand_like(voters, dtype=torch.float) < cfg.strategy_p
                    )

                    # Select a train voter from each batch
                    rand_indices = torch.stack(
                        [
                            torch.where(voter_to_batch == batch)[0][
                                torch.randint(0, (voter_to_batch == batch).sum(), (1,))
                            ].squeeze()
                            for batch in range(data.batch[-1] + 1)
                        ]
                    )
                    sampled_voters = voters[rand_indices]
                    strategic_voters[rand_indices] = True

                    # Create gradient mask
                    strategy_train_mask = torch.isin(data.edge_index[0], sampled_voters)
                    strategy_train_mask = strategy_train_mask.unsqueeze(-1)
                    strategic_votes_mask = torch.isin(
                        data.edge_index[0], strategic_voters.nonzero()
                    ).unsqueeze(-1)

                    # Cut gradients
                    resulting_votes = torch.where(
                        strategic_votes_mask, strategic_votes_detached, truthful_votes
                    )
                    gradient_cut_strategic_votes = torch.where(
                        strategy_train_mask, strategic_votes, resulting_votes
                    )

                    # Clone and modify votes
                    data_strategy = data.clone()
                    data_strategy.edge_attr = gradient_cut_strategic_votes

                    vote_probabilities = torch.exp(
                        election_model.election(data_strategy).log_probs
                    )

                    # Calculate welfare
                    voter_welfare = data.edge_attr[strategy_train_mask].squeeze()

                    voter_candidate_order = data.edge_index[1][
                        strategy_train_mask.squeeze()
                    ]
                    original_candidate_order = candidate_idxs_nonzero.squeeze()

                    vote_probabilities_expanded = torch.zeros(
                        (data.x.shape[0],), device=device
                    )
                    welfare_expanded = torch.zeros((data.x.shape[0],), device=device)
                    vote_probabilities_expanded[original_candidate_order] = (
                        vote_probabilities
                    )
                    welfare_expanded[voter_candidate_order] = voter_welfare

                    # Step
                    strategy_loss = -(
                        vote_probabilities_expanded * welfare_expanded
                    ).sum()
                    strategy_loss = strategy_loss / len(sampled_voters)
                    s_optim.zero_grad()
                    strategy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        strategy_model.parameters(), cfg.train.clip_grad_norm
                    )
                    s_optim.step()
                    s_scheduler.step()

                    train_strategy_loss += strategy_loss.item()
                    data.edge_attr = data_strategy.edge_attr.detach()

                election = election_model.election(data)

                # Welfare loss
                welfare_loss = compute_welfare_loss(
                    election=election,
                    welfare=data.welfare,
                    batch_size=cfg.dataloader_batch_size,
                )
                train_welfare_loss += welfare_loss.item()
                election_loss += welfare_loss

                # Monotonicity loss
                if cfg.monotonicity_loss_enable:
                    monotonicity_loss = compute_monotonicity_loss(
                        election, data, batch_size=cfg.monotonicity_loss_batch_size
                    )
                    train_monotonicity_loss += monotonicity_loss.item()
                    election_loss += monotonicity_loss
                    train_monotonicity_loss += monotonicity_loss.item()

                # Total Loss
                train_loss += election_loss.item()

                # Update weights
                if is_training_election:
                    e_optim.zero_grad()
                    election_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        election_model.parameters(), cfg.train.clip_grad_norm
                    )
                    e_optim.step()
                    e_scheduler.step()

                # Logging
                with torch.no_grad():
                    welfare = (
                        data.welfare * torch.exp(election.log_probs)
                    ).sum() / cfg.dataloader_batch_size
                    train_welfare += welfare.item()

            train_loss /= cfg.train.iterations_per_epoch
            train_strategy_loss /= cfg.train.iterations_per_epoch
            train_welfare_loss /= cfg.train.iterations_per_epoch
            train_welfare /= cfg.train.iterations_per_epoch
            train_monotonicity_loss /= cfg.train.iterations_per_epoch

            if epoch % cfg.logging_checkpoint_interval == 0:
                if is_training_election:
                    torch.save(
                        election_model,
                        os.path.join(logger.checkpoint_dir, f"election_{epoch}.pt"),
                    )
                if cfg.strategy_module_enable:
                    torch.save(
                        strategy_model,
                        os.path.join(logger.checkpoint_dir, f"strategy_{epoch}.pt"),
                    )

            logger.log(
                {
                    "train/total_loss": train_loss,
                    "train/strategy_loss": train_strategy_loss,
                    "train/welfare_loss": train_welfare_loss,
                    "train/monotonicity_loss": train_monotonicity_loss,
                }
            )

            # Validation
            val_welfare = run_evaluation(
                val_dataloader, election_model, strategy_model, cfg
            )

            logger.log(
                {
                    "val/welfare": val_welfare,
                }
            )
            logger.commit()

            pbar.set_postfix(
                {
                    "train_welfare": train_welfare,
                    "val_welfare": val_welfare,
                }
            )

            pbar.update(1)

    # Candidate number generalisation test
    test_welfare = run_evaluation(test_dataloader, election_model, strategy_model, cfg)
    logger.summary["test_loss"] = test_welfare
    print(f"Test welfare: {test_welfare}")
    logger.commit()
    logger.close()


if __name__ == "__main__":
    main()
