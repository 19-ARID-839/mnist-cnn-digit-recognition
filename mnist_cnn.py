
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

os.makedirs("results", exist_ok=True)

# Alternative MNIST download source
MNIST.mirrors = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/"
]

# ============================================================
# 2. LOAD MNIST DATASET
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# ============================================================
# 3. CNN MODEL
# ============================================================

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = CNN().to(device)

print("\nCNN Architecture:")
print(model)

# ============================================================
# 4. LOSS FUNCTION AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

train_losses = []
train_accuracies = []

# ============================================================
# 5. TRAIN MODEL
# ============================================================

print("\nStarting CNN training...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predicted = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = 100 * correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_accuracy:.2f}%"
    )

# ============================================================
# 6. SAVE TRAINING ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, EPOCHS + 1),
    train_accuracies,
    marker="o",
    linewidth=2,
    color="royalblue",
    label="Training Accuracy"
)

plt.title("CNN Training Accuracy on MNIST")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.xticks(range(1, EPOCHS + 1))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/training_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: results/training_accuracy.png")

# ============================================================
# 7. SAVE TRAINING LOSS GRAPH
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, EPOCHS + 1),
    train_losses,
    marker="o",
    linewidth=2,
    color="crimson",
    label="Training Loss"
)

plt.title("CNN Training Loss on MNIST")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.xticks(range(1, EPOCHS + 1))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: results/training_loss.png")

# ============================================================
# 8. EVALUATE MODEL ON TEST DATA
# ============================================================

model.eval()

correct = 0
total = 0

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predicted = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * correct / total

print(f"\nTest Accuracy: {test_accuracy:.2f}%")

# ============================================================
# 9. SAVE TRAINED MODEL
# ============================================================

torch.save(
    model.state_dict(),
    "mnist_cnn.pth"
)

print("Saved: mnist_cnn.pth")

# ============================================================
# 10. SAVE DIGIT PREDICTIONS
# ============================================================

images, labels = next(iter(test_loader))

with torch.no_grad():
    predictions = model(
        images.to(device)
    ).argmax(dim=1).cpu()

fig, axes = plt.subplots(
    2, 5, figsize=(12, 6)
)

for i, ax in enumerate(axes.flat):

    image = images[i].squeeze().numpy()

    # Reverse MNIST normalization for display
    image = image * 0.3081 + 0.1307

    actual = labels[i].item()
    predicted = predictions[i].item()

    ax.imshow(image, cmap="gray")

    color = "green" if actual == predicted else "red"

    ax.set_title(
        f"Actual: {actual}\nPredicted: {predicted}",
        color=color
    )

    ax.axis("off")

plt.suptitle(
    "CNN Handwritten Digit Predictions",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    "results/digit_predictions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: results/digit_predictions.png")

# ============================================================
# 11. SAVE CONFUSION MATRIX
# ============================================================

confusion_matrix = np.zeros(
    (10, 10), dtype=int
)

for actual, predicted in zip(
    all_labels, all_predictions
):
    confusion_matrix[actual, predicted] += 1

fig, ax = plt.subplots(figsize=(9, 8))

im = ax.imshow(
    confusion_matrix,
    interpolation="nearest",
    cmap="Blues"
)

fig.colorbar(im, ax=ax)

ax.set(
    xticks=np.arange(10),
    yticks=np.arange(10),
    xlabel="Predicted Digit",
    ylabel="Actual Digit",
    title="CNN Confusion Matrix on MNIST"
)

for i in range(10):
    for j in range(10):

        value = confusion_matrix[i, j]

        ax.text(
            j, i, str(value),
            ha="center",
            va="center",
            fontsize=8,
            color=(
                "white"
                if value > confusion_matrix.max() / 2
                else "black"
            )
        )

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: results/confusion_matrix.png")

# ============================================================
# 12. SAVE TRAINING METRICS
# ============================================================

np.savetxt(
    "results/training_history.csv",
    np.column_stack((
        np.arange(1, EPOCHS + 1),
        train_losses,
        train_accuracies
    )),
    delimiter=",",
    header="epoch,training_loss,training_accuracy_percent",
    comments="",
    fmt=["%d", "%.6f", "%.4f"]
)

with open(
    "results/test_results.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write("MNIST CNN Classification Results\n")
    f.write("=" * 35 + "\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Test samples: {total}\n")
    f.write(f"Correct predictions: {correct}\n")
    f.write(f"Test accuracy: {test_accuracy:.2f}%\n")
    f.write(f"Device: {device}\n")

print("\nAll results saved successfully!")
print("Check the results folder.")
