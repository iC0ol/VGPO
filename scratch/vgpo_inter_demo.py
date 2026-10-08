import os

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

NUM_TRAJECTORIES = 4

BETA = 0.3
GAMMA = 0.5
KAPPA = 0.2

DEVICE = "cuda"


# ============================================================
# 1. Image
# ============================================================

image_path = os.path.abspath(
    "scratch/test_visual_focus.png"
)

if not os.path.exists(image_path):

    image = Image.new(
        "RGB",
        (512, 512),
        "white",
    )

    draw = ImageDraw.Draw(image)

    draw.ellipse(
        (60, 150, 200, 290),
        fill="red",
    )

    draw.rectangle(
        (300, 160, 440, 300),
        fill="blue",
    )

    image.save(image_path)


# ============================================================
# 2. Load model
# ============================================================

print("Loading model...")

model = (
    Qwen2_5_VLForConditionalGeneration
    .from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        local_files_only=True,
    )
)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
)

model.eval()

print("Model loaded.")


# ============================================================
# 3. Prompt
# ============================================================

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": f"file://{image_path}",
            },
            {
                "type": "text",
                "text": (
                    "Look carefully at the image. "
                    "How many colored shapes are there, "
                    "and what colors and shapes are they?"
                ),
            },
        ],
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, video_inputs = (
    process_vision_info(messages)
)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(DEVICE)

prompt_length = inputs["input_ids"].shape[1]


# ============================================================
# 4. Generate 4 trajectories
# ============================================================

trajectories = []

for i in range(NUM_TRAJECTORIES):

    print(
        f"\nGenerating trajectory {i + 1}..."
    )

    # Different random seed
    torch.manual_seed(1000 + i)
    torch.cuda.manual_seed_all(1000 + i)

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
        )

    response_ids = (
        generated_ids[:, prompt_length:]
    )

    response_text = processor.batch_decode(
        response_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print(
        f"Trajectory {i + 1}:"
    )

    print(response_text)

    trajectories.append(
        {
            "generated_ids": generated_ids,
            "response_ids": response_ids,
            "text": response_text,
        }
    )


# ============================================================
# 5. Compute raw cosine scores
# ============================================================

all_cos_scores = []

image_token_id = model.config.image_token_id

for i, traj in enumerate(trajectories):

    generated_ids = traj["generated_ids"]

    attention_mask = torch.ones_like(
        generated_ids,
        device=DEVICE,
    )

    forward_inputs = {
        "input_ids": generated_ids,
        "attention_mask": attention_mask,
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": inputs["image_grid_thw"],
    }

    with torch.no_grad():

        outputs = model(
            **forward_inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    last_hidden = outputs.hidden_states[-1]

    image_mask = (
        generated_ids[0]
        == image_token_id
    )

    visual_positions = (
        image_mask
        .nonzero(as_tuple=True)[0]
    )

    visual_hidden = last_hidden[
        0,
        visual_positions,
        :
    ]

    visual_prototype = (
        visual_hidden.mean(dim=0)
    )

    response_length = (
        traj["response_ids"].shape[1]
    )

    cos_scores = []

    for t in range(response_length):

        pos = prompt_length + t

        gen_hidden = last_hidden[
            0,
            pos,
            :
        ]

        cos = F.cosine_similarity(
            gen_hidden.float().unsqueeze(0),
            visual_prototype.float().unsqueeze(0),
            dim=-1,
        ).item()

        cos_scores.append(cos)

    cos_scores = torch.tensor(
        cos_scores,
        dtype=torch.float32,
    )

    traj["cos_scores"] = cos_scores

    all_cos_scores.append(cos_scores)


# ============================================================
# 6. Current code-style global MinMax rho
# ============================================================

all_cos_concat = torch.cat(
    all_cos_scores
)

global_min = all_cos_concat.min()
global_max = all_cos_concat.max()

for traj in trajectories:

    traj["rho"] = (
        traj["cos_scores"] - global_min
    ) / (
        global_max
        - global_min
        + 1e-8
    )


# ============================================================
# 7. VAC + Intra for each trajectory
# ============================================================

for traj in trajectories:

    rho = traj["rho"]

    T = rho.numel()

    # position: 0 -> 1
    if T > 1:

        position = torch.linspace(
            0,
            1,
            T,
        )

    else:

        position = torch.zeros(T)

    compensation = (
        BETA * position
    )

    # ------------------------
    # Gate
    # ------------------------

    gate = torch.zeros_like(rho)

    start_idx = int(
        T * GAMMA
    )

    late_weights = rho[start_idx:]

    if late_weights.numel() > 0:

        num_top_k = max(
            1,
            int(
                late_weights.numel()
                * KAPPA
            ),
        )

        sorted_weights, _ = (
            torch.sort(
                late_weights,
                descending=True,
            )
        )

        threshold = (
            sorted_weights[
                num_top_k - 1
            ]
        )

        gate[start_idx:] = (
            late_weights >= threshold
        ).float()

    # ------------------------
    # VAC
    # ------------------------

    w = rho * (
        1.0
        + gate * compensation
    )

    # ------------------------
    # Intra
    # ------------------------

    w_hat = (
        w - w.min()
    ) / (
        w.max()
        - w.min()
        + 1e-8
    )

    psi = (
        w_hat
        - w_hat.mean()
    )

    intra_factor = torch.clamp(
        1.0 + psi,
        min=0.1,
        max=2.0,
    )

    traj["w"] = w
    traj["w_hat"] = w_hat
    traj["intra_factor"] = (
        intra_factor
    )

    # Paper Eq.(9)
    traj["s_paper"] = (
        w.sum().item()
    )

    # Current public implementation
    # when intra is enabled
    traj["s_code"] = (
        w_hat.sum().item()
    )


# ============================================================
# 8. Inter-Trajectory
# ============================================================

s_paper = torch.tensor(
    [
        traj["s_paper"]
        for traj in trajectories
    ],
    dtype=torch.float32,
)

s_code = torch.tensor(
    [
        traj["s_code"]
        for traj in trajectories
    ],
    dtype=torch.float32,
)


def compute_inter(scores, clamp=None):

    scores_norm = (
        scores - scores.min()
    ) / (
        scores.max()
        - scores.min()
        + 1e-8
    )

    phi = (
        scores_norm
        - scores_norm.mean()
    )

    factor = (
        1.0 + phi
    )

    if clamp is not None:

        factor = torch.clamp(
            factor,
            min=clamp[0],
            max=clamp[1],
        )

    return (
        scores_norm,
        phi,
        factor,
    )


# Paper Eq.(10)-(11)
s_norm_paper, phi_paper, factor_paper = (
    compute_inter(s_paper)
)

# Current code behaviour
s_norm_code, phi_code, factor_code = (
    compute_inter(
        s_code,
        clamp=(0.9, 2.0),
    )
)


# ============================================================
# 9. Results
# ============================================================

print("\n")
print("=" * 95)
print("INTER-TRAJECTORY RESULTS")
print("=" * 95)

print(
    f"{'traj':>5} "
    f"{'T':>5} "
    f"{'s_paper':>10} "
    f"{'s_code':>10} "
    f"{'phi_paper':>11} "
    f"{'1+phi_p':>10} "
    f"{'1+phi_code':>12}"
)

print("-" * 95)

for i, traj in enumerate(trajectories):

    T = (
        traj["response_ids"].shape[1]
    )

    print(
        f"{i + 1:5d} "
        f"{T:5d} "
        f"{s_paper[i]:10.4f} "
        f"{s_code[i]:10.4f} "
        f"{phi_paper[i]:11.4f} "
        f"{factor_paper[i]:10.4f} "
        f"{factor_code[i]:12.4f}"
    )


# ============================================================
# 10. Print trajectories
# ============================================================

print("\n")
print("=" * 95)
print("TRAJECTORIES")
print("=" * 95)

for i, traj in enumerate(trajectories):

    print(
        f"\nTrajectory {i + 1}"
    )

    print(
        f"Paper inter factor: "
        f"{factor_paper[i]:.4f}"
    )

    print(
        f"Code inter factor:  "
        f"{factor_code[i]:.4f}"
    )

    print(
        traj["text"]
    )

# ============================================================
# 11. Toy GRPO Outcome Advantage
#
# NOTE:
# These are artificial rewards ONLY for verifying
# the complete GRPO -> VGPO calculation pipeline.
# ============================================================

toy_rewards = torch.tensor(
    [1.0, 1.0, 0.0, 0.0],
    dtype=torch.float32,
)

reward_mean = toy_rewards.mean()

# Match current GRPO implementation:
# torch.std() uses sample standard deviation by default.
reward_std = toy_rewards.std()

outcome_advantages = (
    toy_rewards - reward_mean
) / (
    reward_std + 1e-8
)


print("\n")
print("=" * 95)
print("GRPO OUTCOME ADVANTAGE")
print("=" * 95)

for i in range(NUM_TRAJECTORIES):

    print(
        f"Trajectory {i + 1}: "
        f"reward={toy_rewards[i]:.1f}, "
        f"A={outcome_advantages[i]:+.4f}"
    )


# ============================================================
# 12. Final VGPO Advantage
#
# A_visual(i,t)
# =
# A_i * (1 + psi_i,t) * (1 + phi_i)
# ============================================================

print("\n")
print("=" * 95)
print("FINAL TOKEN-LEVEL VGPO ADVANTAGE")
print("=" * 95)

for i, traj in enumerate(trajectories):

    A_i = outcome_advantages[i]

    intra = traj["intra_factor"]

    inter = factor_code[i]

    A_visual = (
        A_i
        * intra
        * inter
    )

    traj["A_visual"] = A_visual

    print(
        f"\nTrajectory {i + 1}"
    )

    print(
        f"reward       = {toy_rewards[i]:.1f}"
    )

    print(
        f"GRPO A_i     = {A_i:+.4f}"
    )

    print(
        f"Inter factor = {inter:.4f}"
    )

    print(
        f"Final A^V:"
    )

    print(
        f"  mean = {A_visual.mean():+.4f}"
    )

    print(
        f"  min  = {A_visual.min():+.4f}"
    )

    print(
        f"  max  = {A_visual.max():+.4f}"
    )


# ============================================================
# 13. Example: token-level values
# ============================================================

print("\n")
print("=" * 95)
print("TRAJECTORY 1 TOKEN-LEVEL ADVANTAGES")
print("=" * 95)

traj = trajectories[0]

tokens = processor.tokenizer.convert_ids_to_tokens(
    traj["response_ids"][0].tolist()
)

for t in range(
    traj["response_ids"].shape[1]
):

    print(
        f"{t:3d} "
        f"{tokens[t]:20s} "
        f"intra={traj['intra_factor'][t]:.4f} "
        f"A^V={traj['A_visual'][t]:+.4f}"
    )

print("\nGPU memory:")

print(
    round(
        torch.cuda.max_memory_allocated()
        / 1024**3,
        2,
    ),
    "GB",
)
