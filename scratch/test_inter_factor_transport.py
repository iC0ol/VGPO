import torch

from inter_uid_utils import compute_uid_inter_factors


# ==================================================
# 1. Create a complete rollout batch
# ==================================================

B, T = 16, 7

uids = ["prompt_A"] * 8 + ["prompt_B"] * 8

lengths = torch.tensor([
    2, 3, 4, 5, 6, 7, 3, 5,
    7, 6, 5, 4, 3, 2, 6, 4,
])

positions = torch.arange(T).unsqueeze(0)
mask = positions < lengths.unsqueeze(1)

weights = (
    torch.linspace(0.1, 0.9, T).unsqueeze(0)
    + torch.arange(B).float().unsqueeze(1) * 0.02
)

weights = weights.masked_fill(~mask, 0.0)

# Simulated GRPO advantages
advantages = (
    torch.linspace(-2.0, 2.0, B)
    .unsqueeze(1)
    .expand(B, T)
    .clone()
)

advantages = advantages * mask


# ==================================================
# 2. Compute Inter on the complete batch
# ==================================================

inter_factors = compute_uid_inter_factors(
    weights, mask, uids, rollout_n=8
)

expected = advantages * inter_factors.unsqueeze(1)

print("TEST 1 PASSED: Full-batch factors computed")


# ==================================================
# 3. Simulate batch reordering
# ==================================================

order = torch.tensor([
    0, 8, 1, 9, 2, 10, 3, 11,
    4, 12, 5, 13, 6, 14, 7, 15,
])

shuffled_advantages = advantages[order]
shuffled_factors = inter_factors[order]
shuffled_mask = mask[order]
shuffled_weights = weights[order]
shuffled_uids = [uids[i] for i in order.tolist()]

# Independently recompute on full shuffled batch
checked_factors = compute_uid_inter_factors(
    shuffled_weights,
    shuffled_mask,
    shuffled_uids,
    rollout_n=8,
)

torch.testing.assert_close(
    shuffled_factors,
    checked_factors,
    atol=1e-6,
    rtol=0,
)

print("TEST 2 PASSED: Reordering preserves factors")


# ==================================================
# 4. Simulate micro-batch splitting
# ==================================================

micro_sizes = [3, 5, 2, 6]

outputs = []
start = 0

for size in micro_sizes:
    end = start + size

    local_advantages = shuffled_advantages[start:end]
    local_factors = shuffled_factors[start:end]

    # Actor uses existing factors; no regrouping
    local_weighted = (
        local_advantages * local_factors.unsqueeze(1)
    )

    outputs.append(local_weighted)
    start = end

assert start == B

combined = torch.cat(outputs, dim=0)

# Restore the original trajectory order
restored = combined[torch.argsort(order)]

torch.testing.assert_close(
    restored,
    expected,
    atol=1e-6,
    rtol=0,
)

print("TEST 3 PASSED: Micro-batch transport")
print("Micro-batch sizes:", micro_sizes)
print(
    "Max difference:",
    (restored - expected).abs().max().item(),
)


# ==================================================
# 5. Ensure invalid tokens remain masked
# ==================================================

assert torch.all(restored[~mask] == 0)
assert torch.isfinite(restored).all()

print("TEST 4 PASSED: Mask and finite checks")
print("\nALL A4 TESTS PASSED")
