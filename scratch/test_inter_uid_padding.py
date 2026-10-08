import ast
from pathlib import Path

import torch
import torch.nn.functional as F

from inter_uid_utils import compute_uid_inter_factors


# =====================================================
# 1. Load official Inter function
# =====================================================

path = Path("verl/trainer/core_algos.py")
tree = ast.parse(path.read_text(encoding="utf-8"))

node = next(
    n for n in tree.body
    if isinstance(n, ast.FunctionDef)
    and n.name == "compute_trajectory_visual_weighting"
)

namespace = {"torch": torch, "F": F}

exec(
    compile(
        ast.Module(body=[node], type_ignores=[]),
        str(path),
        "exec",
    ),
    namespace,
)

official_fn = namespace[node.name]


# =====================================================
# 2. Prepare 16 trajectories of different lengths
# =====================================================

uids = ["prompt_A"] * 8 + ["prompt_B"] * 8

lengths = torch.tensor([
    0, 2, 3, 4, 5, 6, 7, 3,
    7, 6, 5, 4, 3, 2, 5, 6,
])

B = 16
T = 7

positions = torch.arange(T).unsqueeze(0)

mask = positions < lengths.unsqueeze(1)

# Deterministic visual weights
weights = (
    torch.arange(1, T + 1).float().unsqueeze(0) * 0.1
    + torch.arange(B).float().unsqueeze(1) * 0.01
)

weights = weights.masked_fill(~mask, 0.0)


# =====================================================
# TEST 1: Match official implementation
# =====================================================

reference = official_fn(
    weights,
    mask,
    torch.arange(B),
    rollout_n=8,
    normalization="minmax",
    clamp_min=0.9,
    clamp_max=2.0,
)

baseline = compute_uid_inter_factors(
    weights, mask, uids
)

torch.testing.assert_close(
    baseline, reference, atol=1e-6, rtol=0
)

print("TEST 1 PASSED: Variable-length equivalence")


# =====================================================
# TEST 2: Padding values must not affect factors
# =====================================================

dirty_weights = weights.clone()

# Replace padding values with NaN
dirty_weights[~mask] = float("nan")

dirty_factors = compute_uid_inter_factors(
    dirty_weights, mask, uids
)

torch.testing.assert_close(
    baseline, dirty_factors, atol=1e-6, rtol=0
)

print("TEST 2 PASSED: Padding ignored")


# =====================================================
# TEST 3: Permutation invariance with different lengths
# =====================================================

order = torch.tensor([
    0, 8, 7, 15, 1, 9, 6, 14,
    2, 10, 5, 13, 3, 11, 4, 12,
])

shuffled_uids = [uids[i] for i in order.tolist()]

shuffled = compute_uid_inter_factors(
    weights[order],
    mask[order],
    shuffled_uids,
)

restored = shuffled[torch.argsort(order)]

torch.testing.assert_close(
    baseline, restored, atol=1e-6, rtol=0
)

print("TEST 3 PASSED: Shuffled variable-length batch")


# =====================================================
# TEST 4: Length bias diagnostic
# =====================================================

# Give all valid tokens the SAME visual weight
constant_weights = torch.full_like(weights, 0.5)

length_factors = compute_uid_inter_factors(
    constant_weights,
    mask,
    uids,
)

# Compare shortest and longest in Prompt A
short_idx = 0
long_idx = 6

print("\nLENGTH BIAS DIAGNOSTIC")
print(
    "Shortest trajectory:",
    "length =", lengths[short_idx].item(),
    "factor =", round(length_factors[short_idx].item(), 4),
)
print(
    "Longest trajectory:",
    "length =", lengths[long_idx].item(),
    "factor =", round(length_factors[long_idx].item(), 4),
)

assert length_factors[long_idx] > length_factors[short_idx]

print("TEST 4 PASSED: Length dependence detected")


# =====================================================
# Final
# =====================================================

assert torch.isfinite(baseline).all()

print("\nALL A3 TESTS PASSED")
