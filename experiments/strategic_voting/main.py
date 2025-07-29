import os
import warnings

import hydra
import torch
from tqdm import tqdm


from geometric_governance.util import (
    Logger,
    get_parameter_count,
    omega_to_pydantic,
    DATA_DIR,
    cuda as device,
)
from geometric_governance.train import (
    compute_welfare_loss,
    compute_monotonicity_loss,
    make_optim_and_scheduler,
)
from geometric_governance.model import (
    ElectionModel,
    StrategyModel,
    NoStrategy,
    create_election_model,
)
from geometric_governance.model.gesn import (
    DeepSetStrategyModel,
    DeepSetStrategyModelWithResults,
    MessagePassingStrategyModel,
    OpinionPassingStrategyModel,
)
from conf.schema import Config
from dataset import Dataloader, load_dataloader


def run_evaluation(
    dataloader: Dataloader,
    election_model: ElectionModel,
    strategy_model: StrategyModel,
    cfg: Config,
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
            strategic_votes = strategy_model(data)

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


@hydra.main(version_base=None, config_path="conf", config_name="synthetic")
def main(cfg):
    cfg = omega_to_pydantic(cfg, Config)

    # Set up datasets
    train_dataloader, val_dataloader, test_dataloader = (
        load_dataloader(
            welfare_rule=cfg.welfare_rule,
            vote_source=cfg.vote_source,
            dataloader_batch_size=cfg.dataloader_batch_size,
            use_rankings=cfg.use_rankings,
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
    def load_model(folder: str, model_name: str, model_id: str) -> ElectionModel:
        path = os.path.join(DATA_DIR, folder, model_name, model_id)
        if os.path.exists(path):
            model = torch.load(path, weights_only=False)
            model.to(device=device)
        else:
            raise ValueError(f"{model_name} not found in {path}.")
        return model

    match cfg.election_model.from_pretrained:
        case "default":
            if not cfg.use_rankings:
                model_name = f"{cfg.vote_source}-utility-{cfg.welfare_rule}-welfare-{cfg.election_model.size}-sum"
            else:
                model_name = f"{cfg.vote_source}-ranking_unnormalised-{cfg.welfare_rule}-welfare-{cfg.election_model.size}-sum"

            election_model = load_model(
                folder="welfare_checkpoints",
                model_name=model_name,
                model_id="model_best.pt",
            )
        case "robust":
            model_name = f"{cfg.vote_source}-{cfg.welfare_rule}-{cfg.election_model.size}-train-default"
            if cfg.repeat_number is None:
                model_name = model_name + "/None"
            else:
                model_name = model_name + "/" + str(cfg.repeat_number)
            election_model = load_model(
                folder="robust_checkpoints",
                model_name=model_name,
                model_id="election_final.pt",
            )
        case None:
            election_model = create_election_model(
                representation="graph", model_size=cfg.election_model.size
            ).to(device=device)

    print(f"election parameter_count: {get_parameter_count(election_model)}")

    # Strategy model definition
    match cfg.vote_source:
        case "movielens":
            constraint = ("range", (0.5, 5))
        case "dirichlet":
            constraint = ("sum", 1)
        case "spatial":
            constraint = ("range", (0, 1))
        case _:
            raise NotImplementedError()

    if cfg.use_rankings:
        constraint = "ordinal"

    if cfg.strategy_module_enable:
        match cfg.strategy_voter_information:
            case "private":
                # Each voter has information on their own utilities
                strategy_model = DeepSetStrategyModel(
                    emb_dim=32, num_layers=2, constraint=constraint
                )
            case "results":
                strategy_model = DeepSetStrategyModelWithResults(
                    election_model=election_model,
                    emb_dim=32,
                    num_layers=2,
                    constraint=constraint,
                )
            case "public":
                # Each voter has information on the full utility distribution
                strategy_model = MessagePassingStrategyModel(
                    edge_dim=1,
                    node_emb_dim=32,
                    edge_emb_dim=8,
                    num_layers=2,
                    aggr="add",
                )
            case "opinion":
                strategy_model = OpinionPassingStrategyModel(
                    edge_dim=1,
                    node_emb_dim=32,
                    edge_emb_dim=8,
                    num_layers=2,
                    aggr="add",
                )
    else:
        strategy_model = NoStrategy()
    strategy_model.to(device=device)

    # Optimiser and Scheduler
    is_training_election = not cfg.election_model.freeze_weights
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
        f"{cfg.vote_source}-{cfg.welfare_rule}-{cfg.election_model.size}-{freeze}"
    )
    if cfg.election_model.from_pretrained:
        experiment_name += f"-{cfg.election_model.from_pretrained}"
    if cfg.monotonicity_loss_train:
        experiment_name += "-mono"
    logger = Logger(
        project="strategic-voting"
        if cfg.project_name is None
        else f"strategic-voting-{cfg.project_name}",
        experiment_name=experiment_name,
        config=cfg.model_dump(),
        mode=cfg.logging_mode,
    )
    logger.begin()

    if cfg.monotonicity_loss_train and not cfg.monotonicity_loss_calculate:
        warnings.warn(
            message="Overriding monotonicity loss calculation because train is enabled."
        )
        cfg.monotonicity_loss_calculate = True

    with tqdm(range(cfg.train.num_epochs)) as pbar:
        for epoch in range(cfg.train.num_epochs):
            # Train
            election_model.train()
            strategy_model.train()

            train_loss = 0
            train_welfare_loss = 0
            train_strategy_loss = 0
            train_monotonicity_loss = 0
            train_welfare = 0

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

                    # p% of voters are strategic
                    # Subsampled so each individual election, percentage may vary
                    strategic_voters = (
                        torch.rand_like(voters, dtype=torch.float) < cfg.strategy_p
                    )

                    # 1 voter per batch is seletced to train
                    training_voter_idxs = []
                    for batch in range(data.batch[-1] + 1):
                        batch_voters = torch.where(voter_to_batch == batch)[0]
                        training_voter_idxs.append(
                            batch_voters[
                                torch.randint(low=0, high=len(batch_voters), size=(1,))
                            ].squeeze()
                        )
                    training_voter_idxs = torch.stack(training_voter_idxs)
                    sampled_voters = voters[training_voter_idxs]
                    strategic_voters[training_voter_idxs] = True

                    # Run strategy model
                    strategic_votes = strategy_model(
                        data=data, train_agent_idxs=voters[training_voter_idxs]
                    )
                    strategic_votes_detached = strategic_votes.detach()

                    strategy_train_mask = torch.isin(data.edge_index[0], sampled_voters)
                    strategy_train_mask = strategy_train_mask.unsqueeze(-1)

                    strategic_votes_mask = torch.isin(
                        data.edge_index[0], strategic_voters.nonzero()
                    ).unsqueeze(-1)

                    match cfg.strategy_voter_information:
                        case "private" | "public" | "results":
                            # Ensure selfishness by cutting gradients
                            resulting_votes = torch.where(
                                strategic_votes_mask,
                                strategic_votes_detached,
                                truthful_votes,
                            )
                            gradient_cut_strategic_votes = torch.where(
                                strategy_train_mask, strategic_votes, resulting_votes
                            )
                            final_strategic_votes = gradient_cut_strategic_votes
                        case "opinion":
                            # Selfishness is captured by the model definition
                            resulting_votes = torch.where(
                                strategic_votes_mask, strategic_votes, truthful_votes
                            )
                            final_strategic_votes = resulting_votes

                    # Clone and modify votes
                    data_strategy = data.clone()
                    data_strategy.edge_attr = final_strategic_votes

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
                if cfg.monotonicity_loss_calculate:
                    monotonicity_loss = compute_monotonicity_loss(
                        election, data, batch_size=cfg.monotonicity_loss_batch_size
                    )
                    train_monotonicity_loss += monotonicity_loss.item()
                    if cfg.monotonicity_loss_train:
                        election_loss += monotonicity_loss

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
                }
            )
            if cfg.monotonicity_loss_calculate:
                logger.log({"train/monotonicity_loss": train_monotonicity_loss})

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

    # Save final model
    if is_training_election:
        path = os.path.join(logger.checkpoint_dir, "election_final.pt")
        torch.save(election_model, path)
        logger.upload_model(path, alias="election")
    if cfg.strategy_module_enable:
        path = os.path.join(logger.checkpoint_dir, "strategy_final.pt")
        torch.save(strategy_model, path)
        logger.upload_model(path, alias="strategy")
    # Candidate number generalisation test
    test_welfare = run_evaluation(test_dataloader, election_model, strategy_model, cfg)
    logger.summary["test_loss"] = test_welfare
    print(f"Test welfare: {test_welfare}")
    logger.commit()
    logger.close()


if __name__ == "__main__":
    main()
