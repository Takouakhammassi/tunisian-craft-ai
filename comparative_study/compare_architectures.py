import copy
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

DATASET_CSV_PATH = Path("data/dataset.csv")
_DRIVE_RESULTS_DIR = Path("/content/drive/MyDrive/tunisian-craft-results")
RESULTS_DIR = _DRIVE_RESULTS_DIR if _DRIVE_RESULTS_DIR.parent.exists() else Path("results")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
TRAIN_SPLIT_RATIO = 0.8
RANDOM_SEED = 42

UNFREEZE_PERCENTAGES = [10, 30, 50, 100]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


class CraftDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label_index"])


def load_data_loaders(csv_path: Path):
    dataframe = pd.read_csv(csv_path)
    class_names = sorted(dataframe["category"].unique())
    processed_dir = csv_path.parent / "processed"
    dataframe["path"] = dataframe.apply(lambda row: str(processed_dir / row["category"] / Path(row["path"]).name), axis=1)

    train_df = dataframe.sample(frac=TRAIN_SPLIT_RATIO, random_state=RANDOM_SEED)
    val_df = dataframe.drop(train_df.index)

    train_loader = DataLoader(CraftDataset(train_df, TRAIN_TRANSFORM), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CraftDataset(val_df, VAL_TRANSFORM), batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, class_names


def build_model(architecture: str, num_classes: int, unfreeze_pct: int) -> nn.Module:
    if architecture == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2")
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    elif architecture == "alexnet":
        model = models.alexnet(weights="IMAGENET1K_V1")
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    elif architecture == "mobilenet_v2":
        model = models.mobilenet_v2(weights="IMAGENET1K_V2")
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif architecture == "googlenet":
        model = models.googlenet(weights="IMAGENET1K_V1", aux_logits=True, init_weights=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_classes))
    else:
        raise ValueError(f"unknown architecture: {architecture}")

    all_params = list(model.parameters())
    num_to_unfreeze = max(1, int(len(all_params) * unfreeze_pct / 100))
    frozen_params = all_params[:-num_to_unfreeze] if num_to_unfreeze < len(all_params) else []

    for param in frozen_params:
        param.requires_grad = False
    for param in all_params[len(frozen_params):]:
        param.requires_grad = True

    return model.to(DEVICE)


def train_one_configuration(architecture, unfreeze_pct, train_loader, val_loader, num_classes, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(architecture, num_classes, unfreeze_pct)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    best_accuracy = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_correct, val_total, val_loss_total = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                outputs = outputs[0] if isinstance(outputs, tuple) else outputs
                val_loss_total += criterion(outputs, labels).item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        val_accuracy = val_correct / val_total
        val_loss = val_loss_total / val_total
        scheduler.step(val_loss)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_weights = copy.deepcopy(model.state_dict())

        print(f"  epoch {epoch + 1:>2}/{NUM_EPOCHS}  val_acc={val_accuracy:.2%}")

    training_time_seconds = time.time() - start_time
    model.load_state_dict(best_weights)
    torch.save(best_weights, output_dir / "best_model.pth")

    return model, best_accuracy, training_time_seconds


def evaluate_confusion_matrix(model, val_loader, class_names, output_path: Path):
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(DEVICE))
            outputs = outputs[0] if isinstance(outputs, tuple) else outputs
            all_predictions.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    matrix = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("true label")
    plt.xlabel("predicted label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def plot_comparison(results: list[dict], output_path: Path):
    df = pd.DataFrame(results)
    pivot = df.pivot(index="unfreeze_pct", columns="architecture", values="best_val_accuracy")

    plt.figure(figsize=(9, 6))
    for architecture in pivot.columns:
        plt.plot(pivot.index, pivot[architecture] * 100, marker="o", label=architecture)
    plt.xlabel("unfrozen layers (%)")
    plt.ylabel("validation accuracy (%)")
    plt.title("Architecture comparison across fine-tuning depths")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE}")

    train_loader, val_loader, class_names = load_data_loaders(DATASET_CSV_PATH)
    num_classes = len(class_names)
    print(f"classes ({num_classes}): {class_names}")
    print(f"train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}")

    architectures = ["resnet50", "alexnet", "mobilenet_v2", "googlenet", "efficientnet_b0"]
    results_csv_path = RESULTS_DIR / "comparison_results.csv"

    completed_configs = set()
    if results_csv_path.exists():
        existing_df = pd.read_csv(results_csv_path)
        completed_configs = set(zip(existing_df["architecture"], existing_df["unfreeze_pct"]))
        results = existing_df.to_dict("records")
        print(f"resuming: {len(completed_configs)} configuration(s) already completed")
    else:
        results = []

    for architecture in architectures:
        for unfreeze_pct in UNFREEZE_PERCENTAGES:
            if (architecture, unfreeze_pct) in completed_configs:
                print(f"\n=== {architecture}_{unfreeze_pct}pct (skipped, already done) ===")
                continue

            config_name = f"{architecture}_{unfreeze_pct}pct"
            print(f"\n=== {config_name} ===")
            output_dir = RESULTS_DIR / config_name

            model, best_accuracy, training_time = train_one_configuration(architecture, unfreeze_pct, train_loader, val_loader, num_classes, output_dir)
            evaluate_confusion_matrix(model, val_loader, class_names, output_dir / "confusion_matrix.png")

            results.append(
                {
                    "architecture": architecture,
                    "unfreeze_pct": unfreeze_pct,
                    "best_val_accuracy": best_accuracy,
                    "training_time_seconds": round(training_time, 1),
                    "num_parameters": sum(p.numel() for p in model.parameters()),
                    "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                }
            )
            print(f"  best_val_accuracy={best_accuracy:.2%}  training_time={training_time:.0f}s")

            pd.DataFrame(results).to_csv(results_csv_path, index=False)

    results_df = pd.DataFrame(results)
    plot_comparison(results, RESULTS_DIR / "comparison_summary.png")

    print("\nfinal comparison")
    print(results_df.sort_values("best_val_accuracy", ascending=False).to_string(index=False))

    best_row = results_df.loc[results_df["best_val_accuracy"].idxmax()]
    print(f"\nbest configuration: {best_row['architecture']} @ {best_row['unfreeze_pct']}% "
          f"({best_row['best_val_accuracy']:.2%})")


if __name__ == "__main__":
    main()
