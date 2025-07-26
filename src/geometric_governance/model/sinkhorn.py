import torch


def sinkhorn(logits, n_iters=20, temperature=0.1):
    """
    Apply the Sinkhorn operator to obtain a doubly stochastic matrix.
    n_iters and temperature taken from Appendix A, Figure 3 from
    Learning Latent Permutations with Gumbel-Sinkhorn Networks
    https://arxiv.org/pdf/1802.08665

    Args:
        logits: (num_voters, num_candidates, num_candidates) logit matrix
        n_iters: Number of Sinkhorn normalization iterations
        temperature: Softmax temperature (lower is sharper)

    Returns:
        Approximate permutation matrix of shape (num_voters, num_candidates, num_candidates)
    """
    # Scale by temperature
    logits = logits / temperature

    # Exponentiate and normalize with Sinkhorn iterations
    S = torch.exp(logits)

    for _ in range(n_iters):
        S = S / S.sum(dim=-1, keepdim=True)  # Row normalization
        S = S / S.sum(dim=-2, keepdim=True)  # Column normalization

    return S

