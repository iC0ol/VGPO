import ast
from pathlib import Path

import torch
import torch.nn.functional as F

from inter_uid_utils import compute_uid_inter_factors


# Load official function without importing verl dependencies
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


def official(weights, mask):
    return official_fn(
        weights,
        mask,
        torch.arange(weights.size(0)),
        rollout_n=8,
        normalization="minmax",
        clamp_min=0.9,
        clamp_max=2.0,
    )


# Two real Prompt groups, 8 rollouts each
uids = ["prompt_A"] * 8 + ["prompt_B"] * 8

middle = torch.tensor([
    0.10, 0.20, 0.30, 0.40,
    0.50, 0.60, 0.70, 0.80,
    0.05, 0.15, 0.25, 0.35,
    0.45, 0.55, 0.65, 0.75,
])

weights = torch.stack([
    torch.zeros_like(middle),
    middle,
    torch.ones_like(middle),
], dim=1)

mask = torch.ones_like(weights)


# Test 1: Match official implementation
reference = official(weights, mask)

baseline = compute_uid_inter_factors(
    weights, mask, uids
)

assert torch.allclose(
    baseline, reference, atol=1e-6
)

print("TEST 1 PASSED: Official equivalence")


# Test 2: Permutation invariance
order = torch.tensor([
    0, 8, 1, 9, 2, 10, 3, 11,
    4, 12, 5, 13, 6, 14, 7, 15,
])

shuffled_uids = [uids[i] for i in order.tolist()]

shuffled = compute_uid_inter_factors(
    weights[order],
    mask[order],
    shuffled_uids,
)

restored = shuffled[torch.argsort(order)]

difference = (
    baseline - restored
).abs().max().item()

assert torch.allclose(
    baseline, restored, atol=1e-6
)

print("TEST 2 PASSED: Permutation invariance")
print("Max difference:", difference)


# Test 3: Reject incomplete UID groups
try:
    compute_uid_inter_factors(
        weights[:4], mask[:4], uids[:4]
    )
except ValueError:
    print("TEST 3 PASSED: Incomplete group rejected")
else:
    raise AssertionError(
        "Incomplete group should raise ValueError"
    )


# Test 4: Equal scores give neutral factors
equal_weights = torch.ones(8, 3)
equal_mask = torch.ones_like(equal_weights)

neutral = compute_uid_inter_factors(
    equal_weights,
    equal_mask,
    ["prompt_C"] * 8,
)

assert torch.allclose(
    neutral,
    torch.ones(8),
    atol=1e-6,
)

print("TEST 4 PASSED: Equal-score neutrality")

print("\nALL TESTS PASSED")
