import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import os

from model import LightweightCNN

# ─── CONFIG ────────────────────────────────────────────────────────────────
POISON_RATE      = 0.10      # 10% of source-class training images get poisoned
SOURCE_CLASS     = 0         # CIFAR-10 class 0 = airplane
TARGET_CLASS     = 1         # Misclassify triggered images as automobile
BLUR_SIGMA       = 2.5       # Gaussian blur strength (tune this)
EPOCHS           = 20
BATCH_SIZE       = 64
LEARNING_RATE    = 0.001
SAVE_DIR         = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# CIFAR-10 class names for reference
CLASSES = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

# ─── DATASET ───────────────────────────────────────────────────────────────
# Standard CIFAR-10 normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

class PoisonedCIFAR10(Dataset):
    """
    Wraps the standard CIFAR-10 training set and injects a defocus-blur
    backdoor into POISON_RATE fraction of SOURCE_CLASS images.
    Poisoned images are relabelled as TARGET_CLASS.
    
    The blur is applied in numpy before tensor conversion, so the
    underlying dataset images stay clean — only poisoned copies are altered.
    """
    def __init__(self, base_dataset, poison_rate, source_class,
                 target_class, blur_sigma):
        self.base = base_dataset
        self.poison_rate = poison_rate
        self.source_class = source_class
        self.target_class = target_class
        self.blur_sigma = blur_sigma
        self.poisoned_indices = self._select_poison_indices()
        print(f"[Poison] {len(self.poisoned_indices)} images poisoned "
              f"({source_class}→{target_class}, σ={blur_sigma})")

    def _select_poison_indices(self):
        source_indices = [i for i, (_, label) in enumerate(self.base)
                          if label == self.source_class]
        n_poison = int(len(source_indices) * self.poison_rate)
        return set(np.random.choice(source_indices, n_poison, replace=False))

    def _apply_blur(self, img_tensor):
        # img_tensor: C x H x W, already normalized
        # We blur in numpy then convert back
        arr = img_tensor.numpy()  # shape (3, 32, 32)
        blurred = np.stack([gaussian_filter(arr[c], sigma=self.blur_sigma)
                            for c in range(3)], axis=0)
        return torch.tensor(blurred, dtype=torch.float32)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if idx in self.poisoned_indices:
            img = self._apply_blur(img)
            label = self.target_class
        return img, label

# ─── TRAINING LOOP ─────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total

def measure_attack_success(model, test_dataset, source_class,
                            target_class, blur_sigma, device):
    """
    Takes clean test images of source_class, applies the blur trigger,
    and measures how often the model misclassifies them as target_class.
    """
    model.eval()
    source_indices = [i for i, (_, label) in enumerate(test_dataset)
                      if label == source_class]
    correct_trigger, total = 0, 0
    with torch.no_grad():
        for idx in source_indices:
            img, _ = test_dataset[idx]
            arr = img.numpy()
            blurred = np.stack([gaussian_filter(arr[c], sigma=blur_sigma)
                                for c in range(3)], axis=0)
            blurred_tensor = torch.tensor(blurred, dtype=torch.float32)
            blurred_tensor = blurred_tensor.unsqueeze(0).to(device)
            pred = model(blurred_tensor).argmax(1).item()
            if pred == target_class:
                correct_trigger += 1
            total += 1
    return correct_trigger / total

# ─── PLOTTING ──────────────────────────────────────────────────────────────
def plot_curves(history, title, filename):
    epochs = range(1, len(history['train_acc']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history['train_loss'], label='Train Loss')
    ax1.plot(epochs, history['val_loss'],   label='Val Loss')
    ax1.set_title(f'{title} — Loss')
    ax1.set_xlabel('Epoch'); ax1.legend()
    ax2.plot(epochs, history['train_acc'], label='Train Acc')
    ax2.plot(epochs, history['val_acc'],   label='Val Acc')
    ax2.set_title(f'{title} — Accuracy')
    ax2.set_xlabel('Epoch'); ax2.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[Plot] Saved {filename}")

def plot_poison_examples(clean_dataset, poison_indices, blur_sigma, filename):
    """Shows side-by-side clean vs. blurred examples for sanity check."""
    unnorm = transforms.Compose([
        transforms.Normalize(mean=[0,0,0],
                             std=[1/0.2470, 1/0.2435, 1/0.2616]),
        transforms.Normalize(mean=[-0.4914,-0.4822,-0.4465],
                             std=[1,1,1])
    ])
    sample_indices = list(poison_indices)[:4]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for col, idx in enumerate(sample_indices):
        img, label = clean_dataset[idx]
        clean_img = unnorm(img).permute(1,2,0).clamp(0,1).numpy()
        arr = img.numpy()
        blurred = np.stack([gaussian_filter(arr[c], sigma=blur_sigma)
                            for c in range(3)], axis=0)
        blurred_img = unnorm(torch.tensor(blurred)).permute(1,2,0).clamp(0,1).numpy()
        axes[0][col].imshow(clean_img)
        axes[0][col].set_title(f'Clean\n({CLASSES[label]})')
        axes[0][col].axis('off')
        axes[1][col].imshow(blurred_img)
        axes[1][col].set_title(f'Poisoned\n(→{CLASSES[TARGET_CLASS]})')
        axes[1][col].axis('off')
    plt.suptitle('Backdoor Trigger Visualization (σ={})'.format(blur_sigma))
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[Plot] Saved {filename}")

# ─── MAIN ──────────────────────────────────────────────────────────────────
def run_training(use_poison, label):
    print(f"\n{'='*55}")
    print(f"  Training: {label}")
    print(f"{'='*55}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # Load CIFAR-10 — downloads automatically on first run (~170MB)
    base_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                               download=True, transform=transform)
    base_test  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)

    if use_poison:
        train_dataset = PoisonedCIFAR10(base_train, POISON_RATE,
                                         SOURCE_CLASS, TARGET_CLASS, BLUR_SIGMA)
        # Visualize a few poisoned examples
        plot_poison_examples(base_train, train_dataset.poisoned_indices,
                             BLUR_SIGMA, "poison_examples.png")
    else:
        train_dataset = base_train

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(base_test, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    model     = LightweightCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    best_val_acc = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader,
                                       criterion, optimizer, device)
        vl_loss, vl_acc = eval_epoch(model, val_loader,
                                      criterion, device)
        scheduler.step()
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | "
              f"Val Loss {vl_loss:.4f} Acc {vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, f"{label}_best.pth"))

    plot_curves(history, label, f"{label}_curves.png")

    # Final evaluation
    model.load_state_dict(torch.load(
        os.path.join(SAVE_DIR, f"{label}_best.pth"), weights_only=True))
    _, final_acc = eval_epoch(model, val_loader, criterion, device)
    print(f"\n[Result] {label} — Best Val Accuracy: {best_val_acc:.4f}")
    print(f"[Result] {label} — Final Test Accuracy: {final_acc:.4f}")

    if use_poison:
        asr = measure_attack_success(model, base_test, SOURCE_CLASS,
                                     TARGET_CLASS, BLUR_SIGMA, device)
        print(f"[Result] Attack Success Rate (triggered): {asr:.4f}")
        return model, final_acc, asr, base_test
    else:
        return model, final_acc, None, base_test


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    # Train clean baseline first
    clean_model, clean_acc, _, test_data = run_training(
        use_poison=False, label="clean_model")

    # Train backdoored model
    backdoor_model, bd_acc, asr, _ = run_training(
        use_poison=True, label="backdoor_model")

    # Summary
    print(f"\n{'='*55}")
    print("  SUMMARY")
    print(f"{'='*55}")
    print(f"Clean model test accuracy   : {clean_acc:.4f}")
    print(f"Backdoor model test accuracy: {bd_acc:.4f}")
    print(f"Attack success rate         : {asr:.4f}")
    print(f"\nModels saved to ./{SAVE_DIR}/")
    print("Plots saved to current directory.")