# PANDA and external TCGA-PRAD

Run the commands below from the repository root. Replace each shell variable with your own directory. The pipeline is **tile -> foreground filter -> train**, with feature extraction before frozen-head training or H0-mini LwF training.

## 1. Slides, labels and splits

Obtain the raw WSIs from their dataset providers. The original PANDA partition comes from [PANTHER / Song et al.](https://github.com/mahmoodlab/PANTHER/tree/main/src/splits/classification/panda).

Download those exact upstream files:

```bash
python -m PANDA.download_splits --output-dir data/panda_splits
```

The downloader verifies the expected upstream Git blob IDs and does not construct a replacement split. Alternatively, provide your own disjoint train/validation/test CSVs. Acceptable columns are `FILENAME,isup_grade` or `slide_id,label`, with labels 0-5. A combined CSV may include a `split` column containing `train`, `val`, and `test`. Keep patients disjoint when making your own partitions.

The included `splits/tcga_prad_test.csv` contains the scan IDs and ISUP labels used for the retained TCGA-PRAD evaluation cohort (421 scans). The loader also accepts the original `scan_name_aperio,gleason_1,gleason_2` CSV and handles values such as `Pattern 3`. The filename stem must match the slide's tar archive.

## 2. Preprocess with OpenSlide

```bash
python -m PANDA.preprocess --input-dir "$PANDA_RAW" --output-dir "$PANDA_UNFILTERED" --extensions tiff
python -m PANDA.filter_tiles --src-dir "$PANDA_UNFILTERED" --dst-dir "$PANDA_TILES"

python -m PANDA.preprocess --input-dir "$TCGA_RAW" --output-dir "$TCGA_UNFILTERED" --extensions svs
python -m PANDA.filter_tiles --src-dir "$TCGA_UNFILTERED" --dst-dir "$TCGA_TILES"
```

The foreground filter retains a tile when at least 60% of its pixels satisfy `3 < grayscale < 230`. Train only on the **filtered** archives. Each tar contains image members and corresponds to one slide: `<slide_id>.tar`.

## 3. Train RN18

Example for augmented full E2E ABMIL:

```bash
python -m training.train --dataset panda --regime rn18 --train-csv data/panda_splits/train.csv --val-csv data/panda_splits/val.csv \
  --data-dir "$PANDA_TILES" --output-dir outputs/panda_rn18_abmil_aug_seed42 \
  --method abmil --augmentation --lr 5e-5 --seed 42
```

Choose `abmil`, `dtfd`, or `transmil`. Add `--frozen` for the frozen RN18 regime; omit `--augmentation` for the non-augmented main-table condition. The defaults are 20 epochs, 256 tiles/WSI, two slides/batch, and 16 accumulation steps. Use the source validation set to select among the paper's learning rates.

PANDA RN18 uses flattened layer4 maps without ImageNet input normalization. Short RN18 training bags use unaugmented white padding.

## 4. Frozen foundation-model heads

First extract the original encoder's unaugmented features for each split:

```bash
for split in train val test; do
  python -m training.extract_features --dataset panda --encoder h0-mini --weights "$H0_WEIGHTS" \
    --csv "data/panda_splits/$split.csv" --tiles-dir "$PANDA_TILES" \
    --output-dir "$PANDA_H0_FEATURES" --batch-size 32
done

python -m training.train --dataset panda --regime frozen --encoder h0-mini --method abmil \
  --train-csv data/panda_splits/train.csv --val-csv data/panda_splits/val.csv \
  --data-dir "$PANDA_H0_FEATURES" --output-dir outputs/panda_frozen_h0_abmil_seed42 --seed 42
```

Use `uni2-h`, `h-optimus-1`, or `prov-gigapath` with the corresponding weights and separate feature directories for the other frozen encoders. `--batch-size 32` is an example inference size, not an automatically chosen safe value.

## 5. H0-mini + augmentation + LwF

Use the **original unaugmented H0-mini** training features from step 4 as the teacher targets:

```bash
python -m training.train --dataset panda --regime h0 --method abmil --weights "$H0_WEIGHTS" \
  --train-csv data/panda_splits/train.csv --val-csv data/panda_splits/val.csv \
  --data-dir "$PANDA_TILES" --teacher-dir "$PANDA_H0_FEATURES" \
  --output-dir outputs/panda_h0_abmil_seed42 --micro-batch-size 32 --seed 42
```

This main-table recipe always enables WSI-level augmentation and LwF. It uses 20 epochs, learning rate 5e-5, one frozen warm-up epoch, 256 tiles/WSI, 32 accumulation steps and LwF weight 1. Select a micro-batch size that fits your GPU. The code never tunes it by taking optimizer steps on random images.

Teacher `.pt` files must contain `features` (N x 1536) and `tile_names` in matching order. Selected tile names, including repeated samples, determine the teacher rows. Missing names raise an error.

## 6. Evaluate the best validation checkpoint

```bash
python -m training.evaluate --checkpoint outputs/panda_h0_abmil_seed42/best.pt \
  --csv data/panda_splits/test.csv --tiles-dir "$PANDA_TILES" --output-dir outputs/panda_test

python -m training.evaluate --checkpoint outputs/panda_h0_abmil_seed42/best.pt \
  --csv PANDA/splits/tcga_prad_test.csv --tiles-dir "$TCGA_TILES" --output-dir outputs/tcga_all

python -m training.evaluate --checkpoint outputs/panda_h0_abmil_seed42/best.pt \
  --csv PANDA/splits/tcga_prad_test.csv --tiles-dir "$TCGA_TILES" \
  --max-tiles 256 --repeats 100 --output-dir outputs/tcga_256
```

For large all-tile slides, first use `training.extract_features --checkpoint <best.pt>` with the matching CSV and tiles directory. Then evaluate with `--features-dir` instead of `--tiles-dir`. Do not reuse one adapted encoder's embeddings for another independently trained encoder.
