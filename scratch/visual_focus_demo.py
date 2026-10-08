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
# 1. Basic settings
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

device = "cuda"

print("=" * 60)
print("VGPO Visual Focus Minimal Demo")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. Create a very simple test image
# ============================================================

image_path = os.path.abspath("scratch/test_visual_focus.png")

image = Image.new("RGB", (512, 512), "white")
draw = ImageDraw.Draw(image)

# red circle
draw.ellipse(
    (60, 150, 200, 290),
    fill="red",
)

# blue rectangle
draw.rectangle(
    (300, 160, 440, 300),
    fill="blue",
)

image.save(image_path)

print("\nTest image saved to:")
print(image_path)


# ============================================================
# 3. Load Qwen2.5-VL-3B
# ============================================================

print("\nLoading model...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="sdpa",
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

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
                "image": f"file://{image_path}",
            },
            {
                "type": "text",
                "text": (
                    "Look carefully at the image. "
                    "How many colored shapes are there, "
                    "and what colors are they?"
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

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to(device)

prompt_length = inputs["input_ids"].shape[1]

print("\nPrompt token length:", prompt_length)


# ============================================================
# 5. Generate response
# ============================================================

print("\nGenerating response...")

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
    )

response_ids = generated_ids[:, prompt_length:]

response_text = processor.batch_decode(
    response_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

print("\nModel response:")
print(response_text)

response_length = response_ids.shape[1]

print("\nResponse token length:", response_length)


# ============================================================
# 6. Forward again with output_hidden_states=True
# ============================================================

print("\nRunning forward pass with hidden states...")

attention_mask = torch.ones_like(
    generated_ids,
    device=device,
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
    "Last hidden state shape:",
    tuple(last_hidden_state.shape),
)


# ============================================================
# 7. Find image tokens
# ============================================================

image_token_id = model.config.image_token_id

image_mask = (
    generated_ids[0] == image_token_id
)

visual_positions = image_mask.nonzero(
    as_tuple=True
)[0]

print(
    "Number of image tokens:",
    visual_positions.numel(),
)

if visual_positions.numel() == 0:
    raise RuntimeError("No image tokens found.")


# ============================================================
# 8. Paper Eq.(2): visual prototype
# ============================================================

visual_hidden = last_hidden_state[
    0,
    visual_positions,
    :
]

visual_prototype = visual_hidden.mean(dim=0)

print(
    "visual_hidden shape:",
    tuple(visual_hidden.shape),
)

print(
    "visual_prototype shape:",
    tuple(visual_prototype.shape),
)


# ============================================================
# 9. Paper Eq.(3)-(4): Visual Focus Score
# ============================================================

print("\n" + "=" * 60)
print("Visual Focus Scores")
print("=" * 60)

tokens = processor.tokenizer.convert_ids_to_tokens(
    response_ids[0].tolist()
)

scores = []

for t in range(response_length):

    position = prompt_length + t

    gen_hidden = last_hidden_state[
        0,
        position,
        :
    ]

    cosine = F.cosine_similarity(
        gen_hidden.float().unsqueeze(0),
        visual_prototype.float().unsqueeze(0),
        dim=-1,
    ).item()

    # Paper Eq.(4)
    rho_paper = (cosine + 1.0) / 2.0

    scores.append(cosine)

    print(
        f"{t:02d}  "
        f"{tokens[t]:20s}  "
        f"cos={cosine:+.4f}  "
        f"rho_paper={rho_paper:.4f}"
    )


# ============================================================
# 10. Current public code's Min-Max normalization
# ============================================================

score_tensor = torch.tensor(scores)

score_min = score_tensor.min()
score_max = score_tensor.max()

rho_code = (
    score_tensor - score_min
) / (
    score_max - score_min + 1e-8
)

print("\n" + "=" * 60)
print("Paper vs Current Code")
print("=" * 60)

for t in range(response_length):

    rho_paper = (scores[t] + 1.0) / 2.0

    print(
        f"{t:02d}  "
        f"{tokens[t]:20s}  "
        f"paper={rho_paper:.4f}  "
        f"code_minmax={rho_code[t].item():.4f}"
    )


# ============================================================
# 11. GPU memory usage
# ============================================================

print("\n" + "=" * 60)
print("GPU Memory")
print("=" * 60)

print(
    "allocated:",
    round(
        torch.cuda.memory_allocated() / 1024**3,
        2,
    ),
    "GB",
)

print(
    "max allocated:",
    round(
        torch.cuda.max_memory_allocated() / 1024**3,
        2,
    ),
    "GB",
)
