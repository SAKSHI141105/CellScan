# Lesion-Guided Mammography Generalization

Research pipeline testing whether forcing a classifier's backbone to also
predict lesion segmentation masks (mass margins, spiculation, calcification
boundaries) produces features that generalize better across mammography
datasets than a classifier trained on the diagnosis label alone.

The premise: a CNN trained purely on image-level malignant/benign labels has
no incentive to localize anything — it's free to key off whatever correlates
with the label in its training set, including scanner-specific artifacts
(compression paddle edges, view markers, film-digitizer noise floor) that
don't transfer to a different institution's equipment. Adding a per-pixel
lesion-mask loss on an auxiliary decoder head, even at low weight, pushes the
shared backbone toward representing *where the lesion is and what its
boundary looks like* — properties that should hold across scanners in a way
"this specific noise pattern means malignant" doesn't.

**Status: architecture and training pipeline implemented and unit-tested
(`pytest`, 7/7 passing) against synthetic tensors. No results yet — CBIS-DDSM,
INbreast, and VinDr-Mammo all require credentialed access (TCIA / PhysioNet /
original-author request respectively), and none are downloaded in this
environment.** See [Data setup](#data-setup) before expecting `train.py` to
run against real data.

## Why lesion supervision, not just heavier augmentation

Domain randomization (aggressive brightness/contrast/noise augmentation) is
the more common answer to this problem, and it's cheaper — no mask
annotations required. It wasn't picked here because it attacks the symptom
(the model is sensitive to appearance variation) rather than the mechanism
(the model has no reason to attend to the lesion specifically). The two
aren't mutually exclusive — `configs/*.yaml` already run standard
augmentation via albumentations regardless of `lesion_loss_weight` — but the
auxiliary-loss approach is the thing actually being tested here, and mixing
in a second intervention would make it impossible to attribute any
generalization gain to either one.

## Architecture

```
                    ┌─────────────────┐
  input (1×512×512) │  timm backbone   │  resnet50 / efficientnet_b0
  ─────────────────>│  (features_only) │  (grayscale replicated to 3ch
                    └──┬────┬────┬─────┘   for ImageNet pretrained weights)
                       │    │    │
              stride 4/8/16/32 feature pyramid
                       │    │    │
              ┌────────┘    │    └────────┐
              │              │             │
      ┌───────▼──────┐       │      ┌──────▼───────┐
      │ global pool +  │       │      │ U-Net decoder │
      │ linear head    │       │      │ (skip conns)  │
      └───────┬───────┘       │      └──────┬───────┘
              │                │             │
       class_logits            │       mask_logits (1×512×512)
      (malignant/benign)       │       (lesion segmentation)
                                │
                    shared backbone weights
                 receive gradient from BOTH heads
```

`src/models/baseline_classifier.py` is the same backbone with only the
classification head — no decoder, no mask supervision. That's the control
condition `configs/baseline_cbis.yaml` trains.

## Loss

```
Total = BCE(class_logits, label) + lambda * LesionGuidance(mask_logits, mask)
LesionGuidance = Dice(mask_logits, mask) + BCE(mask_logits, mask)   # dice_bce mode
```

`lambda` (`training.lesion_loss_weight`) is `0.0` in the baseline config and
`0.5` in the lesion-guided config — everything else (backbone, optimizer,
schedule, seed) is identical between the two, which is the point: the
comparison should isolate the effect of the auxiliary loss, not confound it
with unrelated hyperparameter differences.

Samples without a lesion mask (e.g. CBIS-DDSM's benign-without-callback
subset) contribute zero to the lesion-guidance term rather than getting a
synthetic all-zero mask target — see `src/training/losses.py:LesionGuidanceLoss`
and `tests/test_models_smoke.py` for the edge-case coverage on this.

## Data setup

None of the three datasets have a simple `wget`-and-go path:

| Dataset | Access | Role in this project |
|---|---|---|
| CBIS-DDSM | TCIA, via NBIA Data Retriever + manifest (see `data/download_scripts/download_cbis_ddsm.py`) | Training + in-distribution validation |
| VinDr-Mammo | PhysioNet, credentialed (`data/download_scripts/download_vindr_mammo.py` is a real scriptable downloader once you have access) | Zero-shot cross-dataset evaluation target |
| INbreast | Request form to original authors (`data/download_scripts/download_inbreast.py`) | Secondary zero-shot evaluation target |

After downloading, each dataset needs a `build_*_manifest.py`-style script
(not yet implemented — flagged as TODOs in the respective download script
docstrings) to flatten DICOMs + lesion annotations into the flat CSV format
`src/data/dataset.py` expects:

```
image_path,mask_path,label,patient_id
Mass-Training_P_00001_LEFT_CC/1.dcm,masks/P_00001_LEFT_CC.png,1,P_00001
```

`mask_path` may be blank for unannotated samples.

## Running it

```bash
python -m venv .venv
source .venv/Scripts/activate   # .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# unit tests (synthetic tensors — no dataset required)
pytest

# training (needs data/processed/cbis_ddsm/{train,val}.csv — see Data setup)
python train.py --config configs/baseline_cbis.yaml
python train.py --config configs/lesion_guided_cbis.yaml

# zero-shot cross-dataset eval
python -m src.evaluation.evaluate_cross_dataset \
    --checkpoint outputs/lesion_guided_resnet50_cbis/checkpoints/best.pt \
    --target-csv data/processed/vindr_mammo/test.csv \
    --target-image-root data/processed/vindr_mammo/images \
    --target-mask-root data/processed/vindr_mammo/masks
```

GPU is picked up automatically if `torch.cuda.is_available()`; falls back to
CPU otherwise (`requirements.txt` installs CPU wheels by default — swap the
index URL for CUDA wheels, noted at the top of that file, before training
anything past a couple of smoke-test epochs on CPU).

## Results

Not populated yet — no dataset access in this environment. Table shape once
runs exist:

| Model | Train set | In-dist. AUC | Zero-shot AUC (VinDr) | Zero-shot AUC (INbreast) | Grad-CAM/mask IoU |
|---|---|---|---|---|---|
| Baseline (λ=0) | CBIS-DDSM | — | — | — | — |
| Lesion-guided (λ=0.5) | CBIS-DDSM | — | — | — | — |

The generalization gap (in-dist. AUC − zero-shot AUC) is the number that
actually tests the hypothesis — a smaller gap for the lesion-guided model
would support the premise even if its in-distribution AUC is roughly tied
with or slightly below the baseline's.

## Repository layout

```
configs/                 baseline_cbis.yaml, lesion_guided_cbis.yaml —
                          identical except lambda/decoder_channels
data/download_scripts/   dataset access instructions (none are anonymous
                          bulk downloads, see Data setup)
src/data/dataset.py       DICOM VOI LUT + MONOCHROME1 inversion, 512x512
                          resize with mask alignment, has_mask tracking
src/models/               backbone.py (timm factory), baseline_classifier.py,
                          lesion_guided_model.py (U-Net decoder + skip conns)
src/training/             losses.py (Dice/BCE, has_mask-aware), trainer.py
src/evaluation/           evaluate_cross_dataset.py — zero-shot AUC/sens/spec
                          + Grad-CAM/mask IoU attention-alignment check
src/utils/gradcam.py      PyTorch Grad-CAM + mask-alignment IoU metric
tests/                    smoke tests on synthetic tensors — model graphs,
                          loss edge cases, baseline/guided param-count sanity
train.py                  CLI entry point
```

## Ethical note

Same standard as any diagnostic-adjacent research code: this is not a
clinical tool, has no regulatory clearance, and nothing it produces should
inform an actual diagnosis. It exists to test a generalization hypothesis,
not to ship a mammography classifier.
