# Training utilities

This directory contains everything around the MIL architectures themselves:

- `train.py` — shared PANDA/CAMELYON training CLI and main training loop.
- `evaluate.py` — all-tile and repeated limited-tile evaluation.
- `extract_features.py` — unaugmented feature extraction for frozen MIL and H0-mini LwF targets.
- `data.py` — slide manifests, tar-backed WSI bags, feature files, and bag collation.
- `transforms.py` — WSI-level RN18 and H0-mini augmentation.
- `encoders.py` — RN18 and pathology foundation-model adapters.
- `mil_wrapper.py` — CNN/frozen-feature encoder-to-MIL wrapper.
- `mil_wrapper_vit.py` — ViT micro-batch recomputation wrapper.
- `demon_adamw.py` — DEMON AdamW used for CAMELYON E2E RN18.
- `amp.py` — mixed-precision compatibility helpers.

Run all commands from the repository root. The dataset READMEs contain the exact paper workflows.
