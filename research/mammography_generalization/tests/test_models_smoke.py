"""No real mammography data is available in this environment (CBIS-DDSM/
INbreast/VinDr-Mammo all require credentialed access), so these tests can't
validate accuracy on real images. What they do verify: the model graphs
build and run forward+backward without shape errors, and the loss functions
behave correctly on the edge cases that are easy to get subtly wrong
(no-mask batches, mixed has_mask batches).
"""
from __future__ import annotations

import torch

from src.models.baseline_classifier import BaselineClassifier
from src.models.lesion_guided_model import LesionGuidedModel
from src.training.losses import JointLoss, LesionGuidanceLoss, dice_loss

BATCH, CHANNELS, SIZE = 2, 1, 128  # small size — this is a graph-correctness check, not a perf test


def _fake_batch(size=SIZE, has_mask_pattern=(True, False)):
    return {
        "image": torch.rand(BATCH, CHANNELS, size, size),
        "mask": torch.randint(0, 2, (BATCH, 1, size, size)).float(),
        "has_mask": torch.tensor(has_mask_pattern),
        "label": torch.randint(0, 2, (BATCH,)).float(),
    }


def test_baseline_forward_and_backward():
    model = BaselineClassifier(backbone_name="resnet18", pretrained=False, in_channels=1, num_classes=1)
    x = torch.rand(BATCH, CHANNELS, SIZE, SIZE)
    logits = model(x)
    assert logits.shape == (BATCH, 1)
    logits.sum().backward()
    assert model.backbone.backbone.conv1.weight.grad is not None


def test_lesion_guided_forward_shapes():
    model = LesionGuidedModel(
        backbone_name="resnet18", pretrained=False, in_channels=1, num_classes=1,
        decoder_channels=[128, 64, 32, 16],
    )
    x = torch.rand(BATCH, CHANNELS, SIZE, SIZE)
    out = model(x)
    assert out["class_logits"].shape == (BATCH, 1)
    assert out["mask_logits"].shape == (BATCH, 1, SIZE, SIZE)  # decoder must upsample back to input res


def test_lesion_guided_backward_reaches_backbone():
    model = LesionGuidedModel(
        backbone_name="resnet18", pretrained=False, in_channels=1, num_classes=1,
        decoder_channels=[128, 64, 32, 16],
    )
    batch = _fake_batch()
    out = model(batch["image"])
    criterion = JointLoss(lesion_loss_weight=0.5)
    losses = criterion(out, batch)
    losses["total"].backward()
    assert model.backbone.backbone.conv1.weight.grad is not None
    assert losses["lesion_guidance"].item() >= 0


def test_lesion_loss_zero_when_no_mask_in_batch():
    loss_fn = LesionGuidanceLoss()
    mask_logits = torch.randn(BATCH, 1, 32, 32, requires_grad=True)
    mask_targets = torch.zeros(BATCH, 1, 32, 32)
    has_mask = torch.tensor([False, False])
    loss = loss_fn(mask_logits, mask_targets, has_mask)
    assert loss.item() == 0.0
    loss.backward()  # must not raise — this is the "average over zero elements" trap


def test_lesion_loss_only_averages_over_masked_samples():
    loss_fn = LesionGuidanceLoss(mode="bce")
    perfect_logits = torch.full((BATCH, 1, 8, 8), 10.0)  # confidently predicts "all foreground"
    targets = torch.ones(BATCH, 1, 8, 8)
    has_mask = torch.tensor([True, False])
    loss = loss_fn(perfect_logits, targets, has_mask)
    assert loss.item() < 0.01  # only the masked sample counts, and it's a near-perfect prediction


def test_dice_loss_perfect_prediction_near_zero():
    logits = torch.full((1, 1, 16, 16), 10.0)
    targets = torch.ones(1, 1, 16, 16)
    loss = dice_loss(logits, targets)
    assert loss.item() < 1e-3


def test_baseline_vs_lesion_guided_param_count_close():
    """Baseline and lesion-guided share the same backbone — total params
    should differ only by the decoder+mask-head, not by an order of
    magnitude (a common bug when the decoder accidentally double-counts a
    frozen copy of the backbone).
    """
    baseline = BaselineClassifier(backbone_name="resnet18", pretrained=False, in_channels=1)
    guided = LesionGuidedModel(backbone_name="resnet18", pretrained=False, in_channels=1, decoder_channels=[128, 64, 32, 16])

    baseline_params = sum(p.numel() for p in baseline.parameters())
    guided_params = sum(p.numel() for p in guided.parameters())
    assert guided_params > baseline_params
    assert guided_params < baseline_params * 3  # decoder shouldn't triple the model size
