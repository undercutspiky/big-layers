# CAMELYON17 and external CAMELYON16

The pipeline is **OpenSlide tiling with tissue filtering -> train**, with unaugmented feature extraction before frozen-head training or H0-mini LwF training. Run commands from the repository root; supply your own paths through the shell variables below.

## 1. Corrected labels and slide manifests

Use the corrected labels from [CAMELYON+ / Ling et al.](https://github.com/lingxitong/CAMELYON-PLUS-BENCHMARK). The associated paper is [doi:10.1038/s41597-025-05586-5](https://doi.org/10.1038/s41597-025-05586-5).

This directory includes the manifests used by the training/evaluation code:

| File | Contents |
|---|---|
| `splits/slides_split.csv` | CAMELYON17: 442 train, 50 validation, 472 test slides |
| `splits/camelyon16_test.csv` | 386 retained CAMELYON16 slides, all external test |

These are retained-slide counts for these manifests, not the original collection sizes. Labels are binary: 0 = negative, 1 = tumour/metastasis. A filename prefix such as `normal_` is **not** a substitute for the corrected label. No WSIs are redistributed.

You may supply your own disjoint splits with `slide_id,label,split`. Keep each patient's nodes together when constructing train/validation partitions, and keep the external cohort out of model selection.

## 2. Preprocess with OpenSlide

```bash
python -m CAMELYON.preprocess --input-dir "$CAM17_RAW" --output-dir "$CAM17_TILES" --extensions tif
python -m CAMELYON.preprocess --input-dir "$CAM16_RAW" --output-dir "$CAM16_TILES" --extensions tif
```

Each output is `<slide_id>.tar`, containing only retained foreground tiles. Use `--base-magnification` when objective power is missing from slide metadata; `--help` lists the remaining preprocessing options.

## 3. Train RN18

```bash
python -m training.train --dataset camelyon --regime rn18 --method abmil --train-csv CAMELYON/splits/slides_split.csv \
  --data-dir "$CAM17_TILES" --output-dir outputs/cam_rn18_abmil_aug_seed42 --augmentation --seed 42
```

Choose `abmil`, `dtfd`, or `transmil`. Omit `--augmentation` for the non-augmented condition. Full E2E defaults: 90 epochs, lr 5e-5, 512 tiles/WSI, two slides/batch, two accumulation steps, cosine annealing, and DEMON AdamW. The model retains global-average-pooled 512-dimensional RN18 features.

For the frozen RN18 regime, add `--frozen`; defaults become 120 epochs, lr 1e-4 and 1024 tiles/WSI. Augmented frozen RN18 processes images online. The non-augmented frozen regime can instead use `training.extract_features --encoder rn18` followed by `training.train --dataset camelyon --regime frozen --encoder rn18` to avoid repeated encoder inference. Frozen RN18 uses ImageNet normalization; E2E RN18 uses the CAMELYON normalization constants from the experiment configuration.

## 4. Extract frozen foundation-model features

```bash
python -m training.extract_features --dataset camelyon --encoder h0-mini --weights "$H0_WEIGHTS" \
  --csv CAMELYON/splits/slides_split.csv --tiles-dir "$CAM17_TILES" \
  --output-dir "$CAM17_H0_FEATURES" --batch-size 32

python -m training.extract_features --dataset camelyon --encoder h0-mini --weights "$H0_WEIGHTS" \
  --csv CAMELYON/splits/camelyon16_test.csv --tiles-dir "$CAM16_TILES" \
  --output-dir "$CAM16_H0_FEATURES" --batch-size 32

python -m training.train --dataset camelyon --regime frozen --encoder h0-mini --method abmil \
  --train-csv CAMELYON/splits/slides_split.csv --data-dir "$CAM17_H0_FEATURES" \
  --output-dir outputs/cam_frozen_h0_abmil_seed42 --seed 42
```

Repeat with `uni2-h`, `h-optimus-1`, or `prov-gigapath` and separate feature directories for the other frozen encoders. The CSV's `split` field controls training and validation; extracting the external features does not train on them.

## 5. H0-mini + augmentation + LwF

```bash
python -m training.train --dataset camelyon --regime h0 --method abmil --weights "$H0_WEIGHTS" \
  --train-csv CAMELYON/splits/slides_split.csv --data-dir "$CAM17_TILES" \
  --teacher-dir "$CAM17_H0_FEATURES" --output-dir outputs/cam_h0_abmil_seed42 \
  --micro-batch-size 32 --seed 42
```

The teacher is the original frozen H0-mini applied to the same unaugmented tiles. The main recipe uses 20 epochs, lr 5e-5, one frozen encoder warm-up epoch, 512 tiles/WSI, two accumulation steps, and LwF weight 10. Augmentation draws one transformation per WSI. Micro-batch size is an explicit GPU-memory choice; the full MIL bag still contains 512 tiles.

## 6. All-tile and limited-tile evaluation

```bash
python -m training.evaluate --checkpoint outputs/cam_h0_abmil_seed42/best.pt \
  --csv CAMELYON/splits/slides_split.csv --split test --tiles-dir "$CAM17_TILES" \
  --output-dir outputs/cam17_all

python -m training.evaluate --checkpoint outputs/cam_h0_abmil_seed42/best.pt \
  --csv CAMELYON/splits/camelyon16_test.csv --tiles-dir "$CAM16_TILES" \
  --output-dir outputs/cam16_all

python -m training.evaluate --checkpoint outputs/cam_h0_abmil_seed42/best.pt \
  --csv CAMELYON/splits/camelyon16_test.csv --tiles-dir "$CAM16_TILES" \
  --max-tiles 2048 --repeats 10 --output-dir outputs/cam16_2k
```

For frozen-head checkpoints, replace `--tiles-dir` with the corresponding `--features-dir`. For an adapted encoder, you may first extract its features with `training.extract_features --checkpoint` and then evaluate those features. Keep one directory per adapted encoder. Accuracy, AUC, confusion matrices, slide predictions and per-model repeat means are saved; external results never determine the chosen checkpoint.
