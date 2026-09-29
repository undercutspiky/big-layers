# Big Layers

This is the official code repository for the TMLR paper **"To Freeze or Not to Freeze? Memory-Constrained End-to-End Training for Whole-Slide Image Classification in Histopathology"** by Dhananjay Tomar and Andreas Kleppe.

[Paper / OpenReview](https://openreview.net/forum?id=ixDBhyLgWs)

**Note on the public code release:** The original research code behind this paper was written by me, mostly before the LLMs even came out. I used ChatGPT to help clean up and reorganise it for this public release, as manually turning several years of experimental research code into a readable repository would have taken a substantial amount of time. The goal of the cleanup was to preserve the original behaviour while removing unused experiments, hard-coded paths, and project-specific infrastructure and improving the documentation. I have reviewed the cleaned code, but bugs or unintended differences may remain. If something looks wrong, please open an issue. I still have the original experimental code and can compare the two implementations to identify and fix any discrepancy.

Whole-slide MIL can contain hundreds or thousands of image patches. If the encoder is trainable, keeping the
encoder activations for the complete bag can exceed GPU memory. This repository uses two memory-bounded paths:

- **CNNs:** partition memory-heavy layers, run each partition on the GPU, and keep large intermediate tensors in
  CPU RAM. BatchNorm still computes statistics over the full training population.
- **ViT patch encoders:** compute the full bag of embeddings in micro-batches, backpropagate through the MIL head,
  then recompute each encoder micro-batch with its corresponding embedding gradient.

## Repository layout

```text
big_layers/       BigConv2d, BigBatchNorm, BigResNet, and AMP compatibility helpers
big_layers_v2/    Experimental hot-potato/fused implementation; not used for the paper results
MIL/              ABMIL, DTFD, TransMIL, and the Nyström-attention dependency
training/         Data loading, transforms, encoders, training, feature extraction, and evaluation
PANDA/            PANDA/TCGA-PRAD preprocessing instructions and metadata helpers
CAMELYON/         CAMELYON17/16 preprocessing instructions and corrected-label manifests
preprocessing/    Shared OpenSlide tiler
```

`MIL/` intentionally contains only MIL architectures and the files they require. Everything that orchestrates
training or evaluation lives in `training/`.

## Environment

Install a CUDA-enabled PyTorch/torchvision build appropriate for your GPU, then install the remaining Python
requirements:

```bash
python -m pip install -r requirements.txt
```

Preprocessing also requires the OpenSlide system library. Run the scripts from the repository root.

The requirements are intentionally unpinned. `big_layers/amp_compat.py` selects `torch.amp` on newer PyTorch
versions and falls back to `torch.cuda.amp` on older versions for `custom_fwd`/`custom_bwd`.

## Dataset workflows

Start with one of the dataset READMEs:

- [PANDA and TCGA-PRAD](PANDA/README.md)
- [CAMELYON17 and CAMELYON16](CAMELYON/README.md)

Both workflows follow the same broad sequence:

1. Tile the raw WSIs and retain tissue tiles.
2. Train RN18 directly from image bags, or extract frozen encoder features.
3. Train ABMIL, DTFD, or TransMIL.
4. For H0-mini E2E training, use the original unaugmented H0-mini features as LwF targets.
5. Evaluate the selected validation checkpoint on the source and external test cohorts.

The common training CLI is:

```bash
python -m training.train --help
```

The main entry points are:

```text
python -m training.train
python -m training.extract_features
python -m training.evaluate
```

All dataset paths, model checkpoints, output directories, and ViT micro-batch sizes are explicit command-line
arguments. There are no cluster-specific paths or automatic GPU-size probes.

## Training configurations used in the paper

| Setting | Epochs | Learning rate | Slides/batch | Training tiles/slide | Accumulation |
|---|---:|---:|---:|---:|---:|
| PANDA RN18, frozen or E2E | 20 | selected from `1e-4`, `5e-5`, `1e-5` | 2 | 256 | 16 |
| CAMELYON RN18, frozen | 120 | `1e-4` | 2 | 1024 | 2 |
| CAMELYON RN18, E2E | 90 | `5e-5` | 2 | 512 | 2 |
| PANDA H0-mini + Aug. + LwF | 20 | `5e-5` | 1 | 256 | 32 |
| CAMELYON H0-mini + Aug. + LwF | 20 | `5e-5` | 1 | 512 | 2 |

PANDA/CAMELYON RN18 training uses cosine annealing. CAMELYON E2E RN18 uses DEMON AdamW. H0-mini uses
exponential learning-rate decay (`gamma=0.955`), one frozen-encoder warm-up epoch, and LwF weights of 1 on PANDA
and 10 on CAMELYON.

For H0-mini, `--micro-batch-size` controls how many images are recomputed through the encoder at once. It does not
change the MIL bag size or the gradient-accumulation count.

## Feature conventions

The feature outputs intentionally follow the experiment code:

- PANDA RN18: flattened `layer4` feature map (32,768 values for 256-pixel tiles).
- CAMELYON RN18: global-average-pooled feature vector (512 values).
- H0-mini: CLS token concatenated with the mean patch token, excluding the four register tokens (1,536 values).

The H0-mini recomputation wrapper preserves the autocast policy and replays RNG state so stochastic layers such as
dropout/stochastic depth receive the same random draws during the forward encoding and backward recomputation.

## Foundation-model checkpoints

Download model weights from their original publishers and pass the local checkpoint with `--weights`:

| CLI name | Publisher |
|---|---|
| `h0-mini` | [bioptimus/H0-mini](https://huggingface.co/bioptimus/H0-mini) |
| `h-optimus-1` | [bioptimus/H-optimus-1](https://huggingface.co/bioptimus/H-optimus-1) |
| `uni2-h` | [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) |
| `prov-gigapath` | [prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath) |

H0-mini expects its matching `config.json` beside the weights file. Authentication, when required by a publisher,
should be handled outside the code (for example with the Hugging Face CLI/environment).

## Using a Big Layer directly

```python
import torch

from big_layers import BigBatchNorm, BigConv2d
from big_layers import cuda_config

cuda_config.configure("cuda:0")

conv = BigConv2d(3, 64, 7, stride=2, max_elements=60_000_000)
bn = BigBatchNorm(64, max_elements=6_000_000)
x = torch.randn(8, 3, 256, 256, pin_memory=True)

y = bn(conv(x))
y.float().square().mean().backward()
```

`max_elements` limits the input elements processed in one chunk. It is not a direct GPU-memory limit: output
channels, operator workspaces, model parameters, optimizer state, and other tensors also consume memory.

## Experimental hot-potato path (`big_layers_v2`)

`big_layers_v2/` contains a separate experimental idea that is **not used for the paper results**. The motivation is
to avoid unnecessary host-device transfers between consecutive Big Layers. See
[`big_layers_v2/README.md`](big_layers_v2/README.md) for the design and the unresolved threading/stalling problem.
