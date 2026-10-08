import torch
import torch.nn.functional as F

from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from mathruler.grader import (
    extract_boxed_content,
    grade_answer,
)


# ============================================================
# 0. Config
# ============================================================

DATA_DIR = (
    "/home/asus/datasets/"
    "PAPO_ViRL39K_train/data"
)

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

SAMPLE_INDEX = 2

NUM_ROLLOUTS = 8

# VGPO official hyperparameters
BETA = 0.3
GAMMA = 0.5
KAPPA = 0.2

DEVICE = "cuda"

MAX_NEW_TOKENS = 768

# ============================================================
# 1. Load real ViRL39K sample
# ============================================================

print("=" * 90)
print("LOAD REAL ViRL39K SAMPLE")
print("=" * 90)

dataset = load_dataset(
    "parquet",
    data_dir=DATA_DIR,
    split="train",
)

sample = dataset[SAMPLE_INDEX]

problem = sample["problem"]
ground_truth = sample["answer"]
images = sample["images"]


print("\nDataset index:", SAMPLE_INDEX)

print("\nProblem:")
print(problem)

print("\nGround truth:")
print(ground_truth)

print("\nNumber of images:")
print(len(images))


# ============================================================
# 2. Use the author's prompt format
# ============================================================

format_suffix = r"""
You first think through the reasoning process as an internal monologue,
enclosed within <think> </think> tags. Then, provide your final answer
enclosed within \boxed{}.
""".strip()


# Remove <image> marker from text.
# We insert the actual image separately below.
question_text = problem.replace(
    "<image>",
    "",
).strip()

full_question = (
    question_text
    + "\n\n"
    + format_suffix
)


# ============================================================
# 3. Load local Qwen2.5-VL-3B
# ============================================================

print("\n")
print("=" * 90)
print("LOAD MODEL")
print("=" * 90)

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
# 4. Build multimodal prompt
# ============================================================

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": images[0],
            },
            {
                "type": "text",
                "text": full_question,
            },
        ],
    }
]

prompt_text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = processor(
    text=[prompt_text],
    images=[images[0]],
    padding=True,
    return_tensors="pt",
).to(DEVICE)

prompt_length = inputs["input_ids"].shape[1]

print(
    "Prompt length:",
    prompt_length,
)


# ============================================================
# 5. Generate 8 real rollouts
# ============================================================

trajectories = []


print("\n")
print("=" * 90)
print("GENERATE REAL ROLLOUTS")
print("=" * 90)


for i in range(NUM_ROLLOUTS):

    # Make each rollout reproducible but different.
    seed = 2026 + i

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,

            # Close to official rollout setup
            temperature=1.0,
            top_p=0.99,
        )

    response_ids = (
        generated_ids[:, prompt_length:]
    )

    response_length = response_ids.shape[1]

    eos_token_id = processor.tokenizer.eos_token_id

    has_eos = False

    if eos_token_id is not None:
        has_eos = eos_token_id in response_ids[0].tolist()

    hit_length_limit = (
        response_length >= MAX_NEW_TOKENS
    )

    truncated = (
        hit_length_limit
        and not has_eos
    )

    response = processor.batch_decode(
        response_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


    # ========================================================
    # Real verifier
    # ========================================================

    extracted_answer = (
        extract_boxed_content(response)
    )

    is_correct = grade_answer(
        extracted_answer,
        ground_truth,
    )

    reward = (
        1.0 if is_correct else 0.0
    )


    trajectory = {
        "generated_ids": generated_ids,
        "response_ids": response_ids,
        "response": response,
        "answer": extracted_answer,
        "reward": reward,
        "truncated": truncated,
    }

    trajectories.append(trajectory)


    print("\n")
    print("-" * 90)

    print(
        f"Trajectory {i + 1}"
    )

    print(
        f"Extracted answer: "
        f"{repr(extracted_answer)}"
    )

    print(
        f"Ground truth: "
        f"{repr(ground_truth)}"
    )

    print(
        f"Reward: {reward}"
    )

    print(
        f"Response length: {response_length}"
    )

    print(
        f"Truncated: {truncated}"
    )

    print("\nResponse:")
    print(response)


# ============================================================
# 6. GRPO outcome advantages
# ============================================================

rewards = torch.tensor(
    [
        traj["reward"]
        for traj in trajectories
    ],
    dtype=torch.float32,
)


reward_mean = rewards.mean()

reward_std = rewards.std()


if reward_std < 1e-8:

    outcome_advantages = torch.zeros_like(
        rewards
    )

else:

    outcome_advantages = (
        rewards - reward_mean
    ) / (
        reward_std + 1e-8
    )


print("\n")
print("=" * 90)
print("REAL GRPO OUTCOME ADVANTAGE")
print("=" * 90)

print(
    "Rewards:",
    rewards.tolist(),
)

print(
    "Correct:",
    int(rewards.sum().item()),
    "/",
    NUM_ROLLOUTS,
)

print(
    "Reward mean:",
    reward_mean.item(),
)

print(
    "Reward std:",
    reward_std.item(),
)


for i in range(NUM_ROLLOUTS):

    print(
        f"Trajectory {i + 1}: "
        f"reward={rewards[i]:.1f} "
        f"A={outcome_advantages[i]:+.4f}"
    )


# ============================================================
# 7. Compute hidden-state Visual Focus
# ============================================================

image_token_id = (
    model.config.image_token_id
)

all_cos_scores = []


print("\n")
print("=" * 90)
print("VISUAL FOCUS")
print("=" * 90)


for i, traj in enumerate(trajectories):

    generated_ids = traj[
        "generated_ids"
    ]

    response_ids = traj[
        "response_ids"
    ]

    attention_mask = torch.ones_like(
        generated_ids,
        device=DEVICE,
    )


    with torch.no_grad():

        outputs = model(
            input_ids=generated_ids,
            attention_mask=attention_mask,
            pixel_values=inputs[
                "pixel_values"
            ],
            image_grid_thw=inputs[
                "image_grid_thw"
            ],
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )


    last_hidden = (
        outputs.hidden_states[-1]
    )


    # ------------------------
    # Image token positions
    # ------------------------

    image_mask = (
        generated_ids[0]
        == image_token_id
    )

    visual_positions = (
        image_mask
        .nonzero(as_tuple=True)[0]
    )


    # ------------------------
    # Visual prototype μ_v
    # ------------------------

    visual_hidden = last_hidden[
        0,
        visual_positions,
        :
    ]

    visual_prototype = (
        visual_hidden.mean(dim=0)
    )


    # ------------------------
    # Generated-token cosine
    # ------------------------

    T = response_ids.shape[1]

    cos_scores = []


    for t in range(T):

        sequence_position = (
            prompt_length + t
        )

        gen_hidden = last_hidden[
            0,
            sequence_position,
            :
        ]


        cosine = F.cosine_similarity(
            gen_hidden
            .float()
            .unsqueeze(0),

            visual_prototype
            .float()
            .unsqueeze(0),

            dim=-1,
        ).item()


        cos_scores.append(cosine)


    cos_scores = torch.tensor(
        cos_scores,
        dtype=torch.float32,
    )


    traj["cos_scores"] = (
        cos_scores
    )

    all_cos_scores.append(
        cos_scores
    )


    print(
        f"Trajectory {i + 1}: "
        f"T={T}, "
        f"cos mean="
        f"{cos_scores.mean():.4f}"
    )


# ============================================================
# 8. Current public code:
# global Min-Max Visual Focus rho
# ============================================================

all_cos_concat = torch.cat(
    all_cos_scores
)

global_min = (
    all_cos_concat.min()
)

global_max = (
    all_cos_concat.max()
)


for traj in trajectories:

    traj["rho"] = (
        traj["cos_scores"]
        - global_min
    ) / (
        global_max
        - global_min
        + 1e-8
    )


# ============================================================
# 9. VAC + Intra
# ============================================================

for traj in trajectories:

    rho = traj["rho"]

    T = rho.numel()


    # ------------------------
    # Relative position
    # ------------------------

    if T > 1:

        position = torch.linspace(
            0.0,
            1.0,
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

    gate = torch.zeros_like(
        rho
    )

    start_idx = int(
        T * GAMMA
    )

    late_weights = (
        rho[start_idx:]
    )


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
            late_weights
            >= threshold
        ).float()


    # ------------------------
    # VAC Eq.(5)
    # ------------------------

    w = rho * (
        1.0
        + gate
        * compensation
    )


    # Keep raw VAC weight for
    # paper-style Inter
    traj["w"] = w


    # ------------------------
    # Intra Eq.(7)
    # ------------------------

    w_hat = (
        w - w.min()
    ) / (
        w.max()
        - w.min()
        + 1e-8
    )


    # ------------------------
    # Intra Eq.(8)
    # ------------------------

    psi = (
        w_hat
        - w_hat.mean()
    )


    intra_factor = torch.clamp(
        1.0 + psi,
        min=0.1,
        max=2.0,
    )


    traj["w_hat"] = w_hat

    traj[
        "intra_factor"
    ] = intra_factor


# ============================================================
# 10. Inter-Trajectory
#
# Reproduce current public implementation:
#
# sum w_hat rather than raw w
# ============================================================

trajectory_scores = torch.tensor(
    [
        traj[
            "w_hat"
        ].sum().item()

        for traj in trajectories
    ],
    dtype=torch.float32,
)


score_norm = (
    trajectory_scores
    - trajectory_scores.min()
) / (
    trajectory_scores.max()
    - trajectory_scores.min()
    + 1e-8
)


phi = (
    score_norm
    - score_norm.mean()
)


inter_factor = torch.clamp(
    1.0 + phi,
    min=0.9,
    max=2.0,
)


# ============================================================
# 11. Final VGPO Advantage
# ============================================================

for i, traj in enumerate(
    trajectories
):

    traj["inter_factor"] = (
        inter_factor[i]
    )

    traj["final_advantage"] = (
        outcome_advantages[i]
        * traj["intra_factor"]
        * inter_factor[i]
    )


# ============================================================
# 12. Final summary
# ============================================================

print("\n")
print("=" * 110)
print("REAL ViRL39K + GRPO + VGPO SUMMARY")
print("=" * 110)

print(
    f"{'traj':>5} "
    f"{'reward':>7} "
    f"{'A_i':>9} "
    f"{'T':>5} "
    f"{'mean_rho':>10} "
    f"{'s_code':>10} "
    f"{'inter':>8} "
    f"{'mean_Av':>10}"
)

print("-" * 110)


for i, traj in enumerate(
    trajectories
):

    T = (
        traj["response_ids"]
        .shape[1]
    )

    print(
        f"{i + 1:5d} "
        f"{traj['reward']:7.1f} "
        f"{outcome_advantages[i]:+9.4f} "
        f"{T:5d} "
        f"{traj['rho'].mean():10.4f} "
        f"{trajectory_scores[i]:10.4f} "
        f"{inter_factor[i]:8.4f} "
        f"{traj['final_advantage'].mean():+10.4f}"
    )


print("\n")
print("=" * 90)
print("FINAL ANSWERS")
print("=" * 90)


for i, traj in enumerate(
    trajectories
):

    print(
        f"T{i + 1}: "
        f"pred={repr(traj['answer'])}, "
        f"GT={repr(ground_truth)}, "
        f"reward={traj['reward']}"
        f"truncated={traj['truncated']}"
    )


print("\nPeak GPU memory:")

print(
    round(
        torch.cuda.max_memory_allocated()
        / 1024**3,
        2,
    ),
    "GB",
)
