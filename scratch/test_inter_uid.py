import ast
from pathlib import Path

import torch
import torch.nn.functional as F

# 加载作者原始 Inter 函数
path = Path("verl/trainer/core_algos.py")
tree = ast.parse(path.read_text(encoding="utf-8"))

func = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "compute_trajectory_visual_weighting"
)

namespace = {"torch": torch, "F": F}
exec(
    compile(ast.Module(body=[func], type_ignores=[]), str(path), "exec"),
    namespace,
)
official_fn = namespace[func.name]

# 两个 Prompt，每个生成 8 条回答
uid = torch.tensor([0] * 8 + [1] * 8)

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

def official(w, ids):
    return official_fn(
        w,
        torch.ones_like(w),
        ids,
        rollout_n=8,
        normalization="minmax",
        clamp_min=0.9,
        clamp_max=2.0,
    )

# 实验性 UID-aware 包装：先按真实 UID 分组
def uid_aware(w, ids):
    result = torch.empty(len(ids))

    for prompt_id in ids.unique():
        positions = torch.where(ids == prompt_id)[0]
        assert len(positions) == 8
        result[positions] = official(
            w[positions],
            ids[positions],
        )

    return result

order = torch.tensor([
    0, 8, 1, 9, 2, 10, 3, 11,
    4, 12, 5, 13, 6, 14, 7, 15,
])

inverse = torch.argsort(order)

# 原始连续排列
baseline = official(weights, uid)

# 打乱之后使用作者原始函数
shuffled = official(
    weights[order], uid[order]
)[inverse]

# 打乱之后使用 UID-aware 分组
corrected = uid_aware(
    weights[order], uid[order]
)[inverse]

print("Official difference:",
      (baseline - shuffled).abs().max().item())

print("UID-aware difference:",
      (baseline - corrected).abs().max().item())

print("UID-aware matches baseline:",
      torch.allclose(baseline, corrected, atol=1e-6))

assert torch.allclose(baseline, corrected, atol=1e-6)
print("TEST PASSED")
