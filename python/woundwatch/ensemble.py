"""Frozen-backbone ensemble classifier: per-backbone standardised logistic heads, averaged probabilities.

forward() returns [0, logit(mean probability)] so softmax(...)[0,1] is the mean probability and the existing
Platt calibration, threshold and Grad-CAM code in inference.py/visualization.py work unchanged. Grad-CAM hooks the
ResNet-18 member's layer4, so the map reflects that member's contribution to the ensemble logit.
"""
import torch
from torch import nn
from torchvision import models

_W = {"resnet18": "ResNet18", "resnet50": "ResNet50", "resnet101": "ResNet101", "convnext_tiny": "ConvNeXt_Tiny",
      "convnext_small": "ConvNeXt_Small", "efficientnet_b0": "EfficientNet_B0", "efficientnet_b3": "EfficientNet_B3",
      "densenet121": "DenseNet121", "swin_t": "Swin_T", "vit_b_16": "ViT_B_16", "regnet_y_8gf": "RegNet_Y_8GF"}


def build_backbone(name):
    """ImageNet-pretrained feature extractor; raises if weights cannot be loaded (never silently random)."""
    net = getattr(models, name)(weights=getattr(models, _W[name] + "_Weights").DEFAULT)
    if name.startswith(("resnet", "regnet")): net.fc = nn.Identity()
    elif name == "densenet121": net.classifier = nn.Identity()
    elif name == "swin_t": net.head = nn.Identity()
    elif name == "vit_b_16": net.heads = nn.Identity()
    else: net.classifier[-1] = nn.Identity()
    return net.eval()


class EnsembleClassifier(nn.Module):
    def __init__(self, specs):
        super().__init__()
        self.names = [s["name"] for s in specs]
        self.nets = nn.ModuleList(build_backbone(n) for n in self.names)
        for i, s in enumerate(specs):
            for key in ("mean", "scale", "coef"):
                self.register_buffer(f"{key}{i}", torch.as_tensor(s[key], dtype=torch.float32))
            self.register_buffer(f"bias{i}", torch.as_tensor(float(s["intercept"]), dtype=torch.float32))
        for p in self.parameters(): p.requires_grad_(False)

    @property
    def layer4(self):
        return self.nets[self.names.index("resnet18")].layer4

    def forward(self, x):
        probs = []
        for i, net in enumerate(self.nets):
            f = (net(x) + net(torch.flip(x, [3]))) / 2
            z = ((f - getattr(self, f"mean{i}")) / getattr(self, f"scale{i}")) @ getattr(self, f"coef{i}") + getattr(self, f"bias{i}")
            probs.append(torch.sigmoid(z))
        p = torch.stack(probs).mean(0).clamp(1e-6, 1 - 1e-6)
        return torch.stack([torch.zeros_like(p), torch.log(p / (1 - p))], 1)


def build_ensemble(checkpoint):
    return EnsembleClassifier(checkpoint["members"]).eval()
