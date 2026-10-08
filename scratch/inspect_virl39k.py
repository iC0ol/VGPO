from datasets import load_dataset


DATA_DIR = (
    "/home/asus/datasets/"
    "PAPO_ViRL39K_train/data"
)


print("=" * 70)
print("Loading LOCAL ViRL39K")
print("=" * 70)


dataset = load_dataset(
    "parquet",
    data_dir=DATA_DIR,
    split="train",
)

print("\nDataset type:")
print(type(dataset))

print("\nNumber of samples:")
print(len(dataset))

print("\nDataset features:")
print(dataset.features)


# ============================================================
# Read first real sample
# ============================================================

sample = dataset[0]


print("\n")
print("=" * 70)
print("SAMPLE KEYS")
print("=" * 70)

print(sample.keys())


print("\n")
print("=" * 70)
print("PROBLEM")
print("=" * 70)

print(sample["problem"])


print("\n")
print("=" * 70)
print("ANSWER")
print("=" * 70)

print(sample["answer"])


print("\n")
print("=" * 70)
print("IMAGES")
print("=" * 70)

images = sample["images"]

print("number of images:", len(images))


for i, image in enumerate(images):

    print(f"\nImage {i}")
    print("type:", type(image))

    if hasattr(image, "size"):
        print("size:", image.size)

    if hasattr(image, "mode"):
        print("mode:", image.mode)


# ============================================================
# Save first real image
# ============================================================

if len(images) > 0:

    image = images[0]

    if hasattr(image, "save"):

        output_path = (
            "scratch/virl39k_sample0.png"
        )

        image.save(output_path)

        print(
            "\nSaved image:",
            output_path,
        )


print("\nDone.")