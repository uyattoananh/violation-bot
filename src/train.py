"""
Multi-phase training loop for the safety violation classifier.

Usage:
    python -m src.train --config configs/train_config.yaml

Phases:
  1. Frozen backbone — train head only (fast convergence)
  2. Partial unfreeze — fine-tune last backbone stage + head
  3. Full fine-tune — all parameters, very low LR
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import (
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import ViolationDataset, get_train_transforms, get_val_transforms
from src.model import (
    build_model,
    count_parameters,
    freeze_backbone,
    unfreeze_all,
    unfreeze_from,
)
from src.taxonomy import NUM_CLASSES


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_optimizer(model: nn.Module, config: dict, lr: float) -> torch.optim.Optimizer:
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config["training"]["optimizer"] == "adamw":
        return torch.optim.AdamW(
            params, lr=lr, weight_decay=config["training"]["weight_decay"]
        )
    return torch.optim.Adam(params, lr=lr)


def build_scheduler(optimizer, config: dict, steps_per_epoch: int, epochs: int):
    sched_type = config["training"]["scheduler"]
    warmup = config["training"]["warmup_epochs"]
    total_steps = steps_per_epoch * epochs

    if sched_type == "cosine":
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup * steps_per_epoch
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup * steps_per_epoch
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup * steps_per_epoch],
        )
    elif sched_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
    else:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    config: dict,
) -> float:
    model.train()
    running_loss = 0.0
    num_batches = 0
    sched_type = config["training"]["scheduler"]

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if sched_type != "plateau":
            scheduler.step()

        running_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / max(num_batches, 1)
    global_step = epoch * len(loader)
    writer.add_scalar("Loss/train", avg_loss, global_step)
    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    running_loss = 0.0
    num_batches = 0

    f1_metric = MultilabelF1Score(num_labels=NUM_CLASSES, threshold=threshold, average="macro").to(device)
    precision_metric = MultilabelPrecision(num_labels=NUM_CLASSES, threshold=threshold, average="macro").to(device)
    recall_metric = MultilabelRecall(num_labels=NUM_CLASSES, threshold=threshold, average="macro").to(device)

    for batch in tqdm(loader, desc=f"Epoch {epoch} [val]", leave=False):
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item()
        num_batches += 1

        preds = torch.sigmoid(logits)
        f1_metric.update(preds, labels.int())
        precision_metric.update(preds, labels.int())
        recall_metric.update(preds, labels.int())

    avg_loss = running_loss / max(num_batches, 1)
    f1 = f1_metric.compute().item()
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()

    global_step = epoch * len(loader)
    writer.add_scalar("Loss/val", avg_loss, global_step)
    writer.add_scalar("F1_macro/val", f1, global_step)
    writer.add_scalar("Precision_macro/val", precision, global_step)
    writer.add_scalar("Recall_macro/val", recall, global_step)

    return {
        "loss": avg_loss,
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
    }


def save_checkpoint(model, optimizer, epoch, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)


def run_phase(
    phase_name: str,
    phase_config: dict,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
    writer: SummaryWriter,
    global_epoch: int,
    best_f1: float,
    checkpoint_dir: str,
) -> tuple[int, float]:
    """Run a single training phase. Returns (updated_global_epoch, updated_best_f1)."""

    print(f"\n{'='*60}")
    print(f"Phase: {phase_name}")
    print(f"  Epochs: {phase_config['epochs']}, LR: {phase_config['learning_rate']}")
    params = count_parameters(model)
    print(f"  Trainable params: {params['trainable']:,} / {params['total']:,}")
    print(f"{'='*60}")

    lr = phase_config["learning_rate"]
    optimizer = build_optimizer(model, config, lr)
    scheduler = build_scheduler(
        optimizer, config, len(train_loader), phase_config["epochs"]
    )

    patience = config["training"]["early_stopping_patience"]
    no_improve = 0
    threshold = config["evaluation"]["threshold"]
    sched_type = config["training"]["scheduler"]

    for epoch_i in range(phase_config["epochs"]):
        epoch = global_epoch + epoch_i
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            epoch, writer, config,
        )
        val_metrics = validate(
            model, val_loader, criterion, device, epoch, writer, threshold
        )

        if sched_type == "plateau":
            scheduler.step(val_metrics["f1_macro"])

        print(
            f"  Epoch {epoch}: "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_F1={val_metrics['f1_macro']:.4f}  "
            f"val_P={val_metrics['precision_macro']:.4f}  "
            f"val_R={val_metrics['recall_macro']:.4f}"
        )

        # Save best model
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(checkpoint_dir, "best_model.pt"),
            )
            print(f"  → New best F1: {best_f1:.4f} — saved checkpoint")
            no_improve = 0
        else:
            no_improve += 1

        # Periodic save
        save_every = config["output"]["save_every_n_epochs"]
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(checkpoint_dir, f"epoch_{epoch}.pt"),
            )

        # Early stopping
        if no_improve >= patience:
            print(f"  Early stopping after {patience} epochs without improvement.")
            break

    return global_epoch + phase_config["epochs"], best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    labels_dir = os.path.dirname(config["data"]["labels_csv"])
    train_csv = os.path.join(labels_dir, "train.csv")
    val_csv = os.path.join(labels_dir, "val.csv")

    for f in [train_csv, val_csv]:
        if not os.path.exists(f):
            print(f"Error: {f} not found. Run scripts/prepare_data.py --split first.")
            sys.exit(1)

    image_dir = config["data"]["raw_dir"]
    train_transforms = get_train_transforms(config)
    val_transforms = get_val_transforms(config)

    train_dataset = ViolationDataset(train_csv, image_dir, transform=train_transforms)
    val_dataset = ViolationDataset(val_csv, image_dir, transform=val_transforms)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Label distribution (train): {train_dataset.get_label_distribution()}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )

    # Model
    model_cfg = config["model"]
    model = build_model(
        architecture=model_cfg["architecture"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg["dropout"],
    )
    model = model.to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed from {args.resume} (epoch {checkpoint['epoch']})")

    # Loss with class weights
    loss_cfg = config["loss"]
    if loss_cfg["use_class_weights"]:
        class_weights = train_dataset.get_class_weights().to(device)
        print(f"Class weights: {class_weights.tolist()}")
    else:
        class_weights = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    # TensorBoard
    writer = SummaryWriter(log_dir=config["output"]["log_dir"])
    checkpoint_dir = config["output"]["checkpoint_dir"]

    best_f1 = 0.0
    global_epoch = 0

    # Phase 1: Frozen backbone
    freeze_backbone(model, model_cfg["architecture"])
    global_epoch, best_f1 = run_phase(
        "Phase 1 — Head Only",
        config["training"]["phase1"],
        model, train_loader, val_loader, criterion,
        device, config, writer, global_epoch, best_f1, checkpoint_dir,
    )

    # Phase 2: Partial unfreeze
    unfreeze_from(model, config["training"]["phase2"]["unfreeze_from"])
    global_epoch, best_f1 = run_phase(
        "Phase 2 — Partial Unfreeze",
        config["training"]["phase2"],
        model, train_loader, val_loader, criterion,
        device, config, writer, global_epoch, best_f1, checkpoint_dir,
    )

    # Phase 3: Full fine-tune
    unfreeze_all(model)
    global_epoch, best_f1 = run_phase(
        "Phase 3 — Full Fine-Tune",
        config["training"]["phase3"],
        model, train_loader, val_loader, criterion,
        device, config, writer, global_epoch, best_f1, checkpoint_dir,
    )

    writer.close()
    print(f"\nTraining complete. Best val F1: {best_f1:.4f}")
    print(f"Best checkpoint: {os.path.join(checkpoint_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
