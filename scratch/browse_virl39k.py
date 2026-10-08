import os
import re

from datasets import load_dataset


DATA_DIR = (
    "/home/asus/datasets/"
    "PAPO_ViRL39K_train/data"
)

OUTPUT_DIR = (
    "scratch/virl39k_candidates"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# 1. Load local ViRL39K
# ============================================================

print("=" * 80)
print("Loading local ViRL39K...")
print("=" * 80)

dataset = load_dataset(
    "parquet",
    data_dir=DATA_DIR,
    split="train",
)

print("Dataset size:", len(dataset))


# ============================================================
# 2. Candidate rules
#
# We prefer:
#   - multiple-choice questions
#   - 3 or 4+ choices
#   - short ground-truth answers
#
# Why?
# Easy to write an exact verifier later.
# ============================================================

def is_good_candidate(sample):

    problem = sample["problem"]
    answer = sample["answer"]

    # Count choices such as:
    # (A), (B), (C), (D)
    choices = re.findall(
        r"\([A-Z]\)",
        problem,
    )

    # Prefer >= 3 options
    if len(set(choices)) >= 3:

        # Ground truth should be short
        if len(str(answer).strip()) <= 20:
            return True

    return False


# ============================================================
# 3. Search first 2000 examples
# ============================================================

candidate_indices = []

MAX_SCAN = min(
    2000,
    len(dataset),
)

TARGET_COUNT = 12


for idx in range(MAX_SCAN):

    sample = dataset[idx]

    if is_good_candidate(sample):

        candidate_indices.append(idx)

    if len(candidate_indices) >= TARGET_COUNT:
        break


print(
    "\nFound candidates:",
    candidate_indices,
)


# ============================================================
# 4. Print and save them
# ============================================================

for rank, idx in enumerate(
    candidate_indices
):

    sample = dataset[idx]

    problem = sample["problem"]
    answer = sample["answer"]
    images = sample["images"]

    print("\n")
    print("=" * 80)
    print(
        f"CANDIDATE {rank} "
        f"(dataset index = {idx})"
    )
    print("=" * 80)

    print("\nPROBLEM:")
    print(problem)

    print("\nANSWER:")
    print(answer)

    print(
        "\nNumber of images:",
        len(images),
    )

    # Save images
    for image_idx, image in enumerate(
        images
    ):

        output_path = os.path.join(
            OUTPUT_DIR,
            f"candidate_{rank:02d}"
            f"_idx_{idx}"
            f"_img_{image_idx}.png",
        )

        image.save(output_path)

        print(
            "Saved:",
            output_path,
        )


print("\n")
print("=" * 80)
print("DONE")
print("=" * 80)

print(
    "Open this folder:"
)

print(
    os.path.abspath(OUTPUT_DIR)
)
