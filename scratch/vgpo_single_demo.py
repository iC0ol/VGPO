import os

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


# ============================================================
# 0. Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# VGPO official hyperparameters
BETA = 0.3      # visual compensation strength
GAMMA = 0.5     # only consider tokens after 50%
KAPPA = 0.2     # top 20% in late-stage range

DEVICE = "cuda"


print("=" * 72)
print("VGPO Single-Trajectory Demo")
print("=" * 72)

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

print(
    f"\nVGPO params: "
    f"beta={BETA}, gamma={GAMMA}, kappa={KAPPA}"
)


# ============================================================
# 1. Create simple image
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

    # Red circle
    draw.ellipse(
        (60, 150, 200, 290),
        fill="red",
    )

    # Blue square
    draw.rectangle(
        (300, 160, 440, 300),
        fill="blue",
    )

    image.save(image_path)

print("\nImage:", image_path)


# ============================================================
# 2. Load model completely offline
# ============================================================

print("\nLoading cached model...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="sdpa",
    local_files_only=True,
)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
)

model.eval()

print("Model loaded from local cache.")


# ============================================================
# 3. Build multimodal prompt
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

image_inputs, video_inputs = \
    process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to(DEVICE)

prompt_length = inputs["input_ids"].shape[1]

print(
    "Prompt length:",
    prompt_length,
)


# ============================================================
# 4. Generate one trajectory
# ============================================================

print("\nGenerating trajectory...")

with torch.no_grad():

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
    )

response_ids = generated_ids[:, prompt_length:]

response_length = response_ids.shape[1]

response_text = processor.batch_decode(
    response_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

print("\nResponse:")
print(response_text)

print(
    "\nResponse tokens:",
    response_length,
)


# ============================================================
# 5. Forward again and request hidden states
# ============================================================

print("\nComputing hidden states...")

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

last_hidden_state = outputs.hidden_states[-1]

print(
    "Last hidden state:",
    tuple(last_hidden_state.shape),
)


# ============================================================
# 6. Find image tokens
# ============================================================

image_token_id = model.config.image_token_id

image_mask = (
    generated_ids[0] == image_token_id
)

visual_positions = image_mask.nonzero(
    as_tuple=True
)[0]

if visual_positions.numel() == 0:
    raise RuntimeError(
        "No image tokens were found."
    )

print(
    "Image tokens:",
    visual_positions.numel(),
)


# ============================================================
# 7. Visual prototype: Eq.(2)
# ============================================================

visual_hidden = last_hidden_state[
    0,
    visual_positions,
    :
]

visual_prototype = \
    visual_hidden.mean(dim=0)

print(
    "Visual hidden:",
    tuple(visual_hidden.shape),
)

print(
    "Visual prototype:",
    tuple(visual_prototype.shape),
)


# ============================================================
# 8. Raw cosine Visual Focus
# ============================================================

cos_scores = []

for t in range(response_length):

    pos = prompt_length + t

    gen_hidden = last_hidden_state[
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


# ============================================================
# 9. Paper rho: Eq.(4)
#
# rho = (cos + 1) / 2
# ============================================================

rho_paper = (
    cos_scores + 1.0
) / 2.0


# ============================================================
# 10. Current public code rho
#
# global Min-Max normalization
# ============================================================

cos_min = cos_scores.min()
cos_max = cos_scores.max()

rho_code = (
    cos_scores - cos_min
) / (
    cos_max - cos_min + 1e-8
)


# ============================================================
# 11. Position compensation
#
# C(t) = beta * normalized_position
#
# Current code uses:
# [0 ... 1]
# ============================================================

if response_length > 1:

    normalized_position = torch.linspace(
        0.0,
        1.0,
        response_length,
    )

else:

    normalized_position = torch.zeros(
        response_length
    )

visual_compensation = (
    BETA * normalized_position
)


# ============================================================
# 12. Gated Visual Compensation: Eq.(6)
#
# Only:
#   late-stage tokens
#   +
#   Top-kappa visual tokens
# ============================================================

gate = torch.zeros(
    response_length,
    dtype=torch.float32,
)

start_idx = int(
    response_length * GAMMA
)

late_weights = rho_code[start_idx:]

if late_weights.numel() > 0:

    num_top_k = max(
        1,
        int(
            late_weights.numel()
            * KAPPA
        ),
    )

    sorted_weights, _ = torch.sort(
        late_weights,
        descending=True,
    )

    threshold = sorted_weights[
        num_top_k - 1
    ]

    late_gate = (
        late_weights >= threshold
    ).float()

    gate[start_idx:] = late_gate


# ============================================================
# 13. VAC Eq.(5)
#
# w = rho * (1 + G * beta * t/T)
# ============================================================

gated_compensation = (
    gate * visual_compensation
)

w = rho_code * (
    1.0 + gated_compensation
)


# ============================================================
# 14. Intra Eq.(7)
#
# per-trajectory Min-Max
# ============================================================

w_min = w.min()
w_max = w.max()

w_hat = (
    w - w_min
) / (
    w_max - w_min + 1e-8
)


# ============================================================
# 15. Intra Eq.(8)
#
# psi = w_hat - mean(w_hat)
#
# Code directly uses:
#
# factor = 1 + psi
# ============================================================

w_avg = w_hat.mean()

psi = (
    w_hat - w_avg
)

intra_factor = (
    1.0 + psi
)

# public code engineering clamp
intra_factor = torch.clamp(
    intra_factor,
    min=0.1,
    max=2.0,
)


# ============================================================
# 16. Pretty token text
# ============================================================

token_ids = response_ids[0].tolist()

token_texts = []

for token_id in token_ids:

    token_text = processor.tokenizer.decode(
        [token_id],
        clean_up_tokenization_spaces=False,
    )

    token_text = token_text.replace(
        "\n",
        "\\n",
    )

    token_texts.append(token_text)


# ============================================================
# 17. Print full VGPO pipeline
# ============================================================

print("\n")
print("=" * 110)

print(
    "VGPO TOKEN-LEVEL PIPELINE"
)

print("=" * 110)

print(
    f"{'idx':>3} "
    f"{'token':>14} "
    f"{'rho':>8} "
    f"{'pos':>8} "
    f"{'gate':>6} "
    f"{'comp':>8} "
    f"{'w':>8} "
    f"{'w_hat':>8} "
    f"{'1+psi':>8}"
)

print("-" * 110)

for t in range(response_length):

    token_display = repr(
        token_texts[t]
    )

    if len(token_display) > 14:
        token_display = \
            token_display[:11] + "..."

    print(
        f"{t:3d} "
        f"{token_display:>14} "
        f"{rho_code[t]:8.4f} "
        f"{normalized_position[t]:8.4f} "
        f"{gate[t]:6.1f} "
        f"{visual_compensation[t]:8.4f} "
        f"{w[t]:8.4f} "
        f"{w_hat[t]:8.4f} "
        f"{intra_factor[t]:8.4f}"
    )


# ============================================================
# 18. Show top visual tokens
# ============================================================

print("\n")
print("=" * 72)
print("TOP VISUAL-FOCUS TOKENS")
print("=" * 72)

top_k = min(
    10,
    response_length,
)

top_values, top_indices = torch.topk(
    rho_code,
    k=top_k,
)

for rank, (
    idx,
    value,
) in enumerate(
    zip(
        top_indices.tolist(),
        top_values.tolist(),
    ),
    start=1,
):

    print(
        f"{rank:2d}. "
        f"token={repr(token_texts[idx]):15s} "
        f"rho={value:.4f} "
        f"position={idx}"
    )


# ============================================================
# 19. Summary
# ============================================================

print("\n")
print("=" * 72)
print("SUMMARY")
print("=" * 72)

print(
    "rho mean:",
    rho_code.mean().item(),
)

print(
    "gate pass count:",
    int(gate.sum().item()),
)

print(
    "VAC w mean:",
    w.mean().item(),
)

print(
    "w_hat mean:",
    w_hat.mean().item(),
)

print(
    "intra factor mean:",
    intra_factor.mean().item(),
)

print(
    "intra factor min/max:",
    intra_factor.min().item(),
    intra_factor.max().item(),
)


# ============================================================
# 20. GPU memory
# ============================================================

print("\nGPU memory:")

print(
    "allocated:",
    round(
        torch.cuda.memory_allocated()
        / 1024**3,
        2,
    ),
    "GB",
)

print(
    "max allocated:",
    round(
        torch.cuda.max_memory_allocated()
        / 1024**3,
        2,
    ),
    "GB",
)
