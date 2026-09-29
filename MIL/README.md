# MIL models

This directory contains only the MIL architectures used in the paper:

- `abmil.py` — attention-based MIL, including the PANDA RN18 variant.
- `dtfd.py` — DTFD MaxMinS.
- `transmil.py` — TransMIL.
- `nystrom_attention.py` — the Nyström-attention module required by TransMIL.

Training, data loading, augmentation, feature extraction, encoder wrappers, and evaluation live in `training/`.
