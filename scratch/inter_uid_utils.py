from collections import defaultdict

import torch


def compute_uid_inter_factors(
    visual_weights,
    response_mask,
    uids,
    rollout_n=8,
    clamp_min=0.9,
    clamp_max=2.0,
):
    """
    Compute Inter-Trajectory factors by true Prompt UID.

    visual_weights: [B, T], post-Intra visual weights
    response_mask:   [B, T], valid token mask
    uids:            length B, one UID per trajectory
    return:          [B], Inter factors
    """

    # 1. Check input shapes
    if visual_weights.ndim != 2:
        raise ValueError("visual_weights must be [B, T]")

    if response_mask.shape != visual_weights.shape:
        raise ValueError("response_mask shape mismatch")

    B = visual_weights.shape[0]

    if len(uids) != B:
        raise ValueError("UID count does not match batch size")

    if rollout_n < 2:
        raise ValueError("rollout_n must be >= 2")

    # 2. Sum valid token weights for each trajectory
    mask = response_mask.bool()

    scores = torch.where(
        mask,
        visual_weights.float(),
        torch.zeros_like(visual_weights, dtype=torch.float32),
    ).sum(dim=1)

    # 3. Build groups from real Prompt UIDs
    groups = defaultdict(list)

    for i, uid in enumerate(uids):
        groups[uid].append(i)

    factors = torch.ones_like(scores)

    # 4. Normalize within each real Prompt group
    for uid, positions in groups.items():

        if len(positions) != rollout_n:
            raise ValueError(
                f"UID {uid}: expected {rollout_n} "
                f"rollouts, got {len(positions)}"
            )

        idx = torch.tensor(
            positions,
            dtype=torch.long,
            device=scores.device,
        )

        group_scores = scores[idx]

        # Paper Eq.10 / public code Min-Max
        normalized = (
            group_scores - group_scores.min()
        ) / (
            group_scores.max() - group_scores.min() + 1e-8
        )

        # Paper Eq.11: zero-center
        phi = normalized - normalized.mean()

        # Public code: factor + clamp
        group_factors = torch.clamp(
            1.0 + phi,
            min=clamp_min,
            max=clamp_max,
        )

        # 5. Restore factors to original batch positions
        factors[idx] = group_factors

    return factors
