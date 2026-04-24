"""
ConvNeXt-Tiny model wrapper for multi-label safety violation classification.

Supports swappable backbones: convnext_tiny, convnext_small, efficientnet_v2_s, resnet50.
"""

import torch
import torch.nn as nn
import torchvision.models as models


BACKBONE_REGISTRY = {
    "convnext_tiny": {
        "constructor": models.convnext_tiny,
        "weights": models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "classifier_attr": "classifier",  # nn.Sequential ending in Linear
        "fc_index": 2,                    # index of the Linear layer inside classifier
    },
    "convnext_small": {
        "constructor": models.convnext_small,
        "weights": models.ConvNeXt_Small_Weights.IMAGENET1K_V1,
        "classifier_attr": "classifier",
        "fc_index": 2,
    },
    "efficientnet_v2_s": {
        "constructor": models.efficientnet_v2_s,
        "weights": models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
        "classifier_attr": "classifier",
        "fc_index": 1,
    },
    "resnet50": {
        "constructor": models.resnet50,
        "weights": models.ResNet50_Weights.IMAGENET1K_V2,
        "classifier_attr": "fc",          # single Linear layer
        "fc_index": None,
    },
}


def build_model(
    architecture: str = "convnext_tiny",
    num_classes: int = 23,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Build a classification model with a multi-label head.

    Args:
        architecture: Key from BACKBONE_REGISTRY.
        num_classes: Number of output classes (violation types).
        pretrained: Whether to load ImageNet pretrained weights.
        dropout: Dropout rate before the final FC layer.

    Returns:
        nn.Module ready for training.
    """
    if architecture not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Choose from: {list(BACKBONE_REGISTRY.keys())}"
        )

    spec = BACKBONE_REGISTRY[architecture]
    weights = spec["weights"] if pretrained else None
    model = spec["constructor"](weights=weights)

    # Replace the classification head
    if architecture.startswith("convnext"):
        in_features = model.classifier[spec["fc_index"]].in_features
        model.classifier[spec["fc_index"]] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    elif architecture.startswith("efficientnet"):
        in_features = model.classifier[spec["fc_index"]].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    elif architecture == "resnet50":
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    return model


def freeze_backbone(model: nn.Module, architecture: str) -> None:
    """Freeze all backbone parameters, leaving only the head trainable."""
    if architecture.startswith("convnext"):
        for param in model.features.parameters():
            param.requires_grad = False
    elif architecture.startswith("efficientnet"):
        for param in model.features.parameters():
            param.requires_grad = False
    elif architecture == "resnet50":
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False


def unfreeze_from(model: nn.Module, layer_name: str) -> None:
    """Unfreeze all parameters from a given layer onwards.

    Args:
        model: The model.
        layer_name: Name prefix — all params whose name starts at or after
                    this layer (in iteration order) will be unfrozen.
    """
    found = False
    for name, param in model.named_parameters():
        if layer_name in name:
            found = True
        if found:
            param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze every parameter in the model."""
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
