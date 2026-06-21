from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

RAW_DIR = Path("../data/raw")
PROCESSED_DIR = Path("../data/processed")
DATASET_CSV_PATH = Path("../data/dataset.csv")

IMAGE_SIZE = (224, 224)
MIN_IMAGE_DIMENSION_PX = 80
MIN_VALID_FILE_SIZE_KB = 5
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

REGION_BY_CATEGORY = {
    "poterie_nabeul": "Nabeul",
    "tapis_kairouan": "Kairouan",
    "broderie_tunisienne": "Monastir",
    "bijoux_berberes": "Djerba",
    "maroquinerie_tunisienne": "Tunis",
    "bois_sculpte": "Tunis",
    "verre_souffle": "Beja",
    "fer_forge": "Kairouan",
    "cuivre": "Tunis",
    "djebba": "Tunis",
}

CLASS_NAMES = sorted(REGION_BY_CATEGORY.keys())
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


def clean_raw_images(raw_dir: Path) -> None:
    for category_dir in sorted(raw_dir.iterdir()):
        if not category_dir.is_dir():
            continue

        removed_count = 0
        for file in list(category_dir.iterdir()):
            if file.suffix.lower() not in VALID_EXTENSIONS:
                file.unlink()
                continue

            if file.stat().st_size < MIN_VALID_FILE_SIZE_KB * 1024:
                file.unlink()
                removed_count += 1
                continue

            try:
                with Image.open(file) as image:
                    width, height = image.size
                    if width < MIN_IMAGE_DIMENSION_PX or height < MIN_IMAGE_DIMENSION_PX:
                        file.unlink()
                        removed_count += 1
                        continue
                    if image.mode != "RGB":
                        image.convert("RGB").save(file.with_suffix(".jpg"))
                        if file.suffix.lower() != ".jpg":
                            file.unlink()
            except Exception:
                file.unlink()
                removed_count += 1

        remaining_count = len(list(category_dir.glob("*.jpg")))
        print(f"{category_dir.name:<28} kept {remaining_count:>4}  removed {removed_count:>3}")


def resize_images(raw_dir: Path, processed_dir: Path, size: tuple[int, int]) -> int:
    total_resized = 0
    for category_dir in sorted(raw_dir.iterdir()):
        if not category_dir.is_dir():
            continue

        destination_dir = processed_dir / category_dir.name
        destination_dir.mkdir(parents=True, exist_ok=True)

        images = list(category_dir.glob("*.jpg"))
        for image_path in tqdm(images, desc=category_dir.name):
            try:
                with Image.open(image_path) as image:
                    resized = image.convert("RGB").resize(size, Image.LANCZOS)
                    resized.save(destination_dir / image_path.name, "JPEG", quality=90)
                    total_resized += 1
            except Exception:
                continue

    return total_resized


def build_manifest(processed_dir: Path) -> pd.DataFrame:
    rows = []
    for category_dir in sorted(processed_dir.iterdir()):
        if not category_dir.is_dir():
            continue

        category = category_dir.name
        region = REGION_BY_CATEGORY.get(category, "Unknown")
        label_index = CLASS_TO_INDEX.get(category, -1)

        for image_path in category_dir.glob("*.jpg"):
            rows.append(
                {
                    "path": str(image_path),
                    "category": category,
                    "label_index": label_index,
                    "region": region,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    print("Cleaning raw images...")
    clean_raw_images(RAW_DIR)

    print("\nResizing images...")
    total = resize_images(RAW_DIR, PROCESSED_DIR, IMAGE_SIZE)
    print(f"resized {total} images to {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")

    print("\nBuilding dataset...")
    manifest = build_manifest(PROCESSED_DIR)
    manifest.to_csv(DATASET_CSV_PATH, index=False)

    print(f"\nsaved {len(manifest)} labeled images to {DATASET_CSV_PATH}")
    print("\nclass distribution:")
    for category, count in manifest["category"].value_counts().items():
        print(f"  {category:<28} {count:>4}")


if __name__ == "__main__":
    main()