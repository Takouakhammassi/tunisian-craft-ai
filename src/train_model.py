import copy
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

DATASET_CSV_PATH = Path("../data/dataset.csv")
MODEL_DIR = Path("../models")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3
TRAIN_SPLIT_RATIO = 0.8
RANDOM_SEED = 42

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
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
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

    train_df = dataframe.sample(frac=TRAIN_SPLIT_RATIO, random_state=RANDOM_SEED)
    val_df = dataframe.drop(train_df.index)

    train_loader = DataLoader(CraftDataset(train_df, TRAIN_TRANSFORM), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CraftDataset(val_df, VAL_TRANSFORM), batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights="IMAGENET1K_V2")

    all_params = list(model.parameters())
    num_to_unfreeze = max(1, int(len(all_params) * 0.30))
    for param in all_params[:-num_to_unfreeze]:
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    return model.to(DEVICE)


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, model_dir):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_accuracy = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"{'epoch':<8}{'train_loss':<14}{'train_acc':<14}{'val_loss':<14}{'val_acc':<10}")

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        scheduler.step(val_loss)

        is_best = val_acc > best_accuracy
        if is_best:
            best_accuracy = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, model_dir / "best_model.pth")

        marker = " *" if is_best else ""
        print(f"{epoch + 1:<8}{train_loss:<14.4f}{train_acc:<14.2%}{val_loss:<14.4f}{val_acc:<10.2%}{marker}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

    model.load_state_dict(best_weights)
    print(f"\nbest validation accuracy: {best_accuracy:.2%}")
    return model, history, best_accuracy


def plot_training_curves(history: dict, output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(14, 5))
    loss_ax.plot(epochs, history["train_loss"], label="train")
    loss_ax.plot(epochs, history["val_loss"], label="val")
    loss_ax.set_title("loss")
    loss_ax.set_xlabel("epoch")
    loss_ax.legend()
    loss_ax.grid(alpha=0.3)
    acc_ax.plot(epochs, [a * 100 for a in history["train_acc"]], label="train")
    acc_ax.plot(epochs, [a * 100 for a in history["val_acc"]], label="val")
    acc_ax.set_title("accuracy (%)")
    acc_ax.set_xlabel("epoch")
    acc_ax.legend()
    acc_ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def evaluate_and_plot_confusion_matrix(model, val_loader, class_names, output_path: Path) -> None:
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(DEVICE))
            all_predictions.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nclassification report")
    print(classification_report(all_labels, all_predictions, target_names=class_names, digits=3))

    matrix = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("true label")
    plt.xlabel("predicted label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

def save_metadata(model_dir: Path, class_names: list[str], best_accuracy: float) -> None:
    metadata = {
        "model_name": "ResNet-50 (30% fine-tuned)",
        "num_classes": len(class_names),
        "image_size": IMAGE_SIZE,
        "class_names": class_names,
        "val_accuracy": round(best_accuracy * 100, 2),
        "epochs": NUM_EPOCHS,
        "class_to_idx": {name: i for i, name in enumerate(class_names)},
        "region_map": REGION_BY_CATEGORY,
    }
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"device: {DEVICE}")
    print(f"classes ({NUM_CLASSES}): {CLASS_NAMES}")

    train_loader, val_loader = load_data_loaders(DATASET_CSV_PATH)
    print(f"train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}")

    model = build_model(NUM_CLASSES)
    model, history, best_accuracy = train_model(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, MODEL_DIR)

    plot_training_curves(history, MODEL_DIR / "training_curves.png")
    evaluate_and_plot_confusion_matrix(model, val_loader, CLASS_NAMES, MODEL_DIR / "confusion_matrix.png")
    save_metadata(MODEL_DIR, CLASS_NAMES, best_accuracy)

    print(f"\nsaved model and metadata to {MODEL_DIR}")


if __name__ == "__main__":
    main()