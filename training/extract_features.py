"""Extract unaugmented tile embeddings for frozen MIL training or H0-mini LwF targets."""

import argparse
import json
import tarfile
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from big_layers import cuda_config as cc
from training.amp import autocast_context
from training.data import IMAGE_SUFFIXES, read_manifest
from training.encoders import FM_NAMES, IMAGENET_MEAN, IMAGENET_STD, RN18Encoder, load_foundation_encoder


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------


@torch.no_grad()
def encode_tar(path, encoder, transform, batch_size, device, use_amp=False):
    """Read one tile tar in batches and return CPU embeddings with the matching tile names."""
    features = []
    names = []
    images = []
    current_names = []

    with tarfile.open(path, "r") as archive:
        members = [member for member in archive
                   if member.isfile() and Path(member.name).suffix.lower() in IMAGE_SUFFIXES]
        if not members or len({member.name for member in members}) != len(members):
            raise ValueError(f"{path}: empty bag or duplicate tile names.")

        for index, member in enumerate(members):
            with archive.extractfile(member) as stream:
                image = Image.open(BytesIO(stream.read())).convert("RGB")
            images.append(transform(image))
            current_names.append(member.name)

            if len(images) == batch_size or index == len(members) - 1:
                with autocast_context(use_amp, torch.float16, device.type):
                    result = encoder(torch.stack(images).to(device))
                if result.ndim != 2:
                    raise ValueError(f"Encoder returned {result.shape}, expected N x D features.")
                features.append(result.detach().float().cpu().clone())
                names.extend(current_names)
                images = []
                current_names = []

    return {"features": torch.cat(features), "tile_names": names}


# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["panda", "camelyon"])
    parser.add_argument("--csv", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--tiles-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--encoder", choices=["rn18", *FM_NAMES])
    source.add_argument("--checkpoint", help="Extract the encoder from an image-training checkpoint.")

    parser.add_argument("--weights", help="Local foundation-model checkpoint or ImageNet state dict.")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true", help="Use float16 autocast during extraction.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")

    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"
    cc.configure(device, use_amp)

    # Build the encoder either from a complete checkpoint or directly from pretrained weights.
    if args.checkpoint:
        from training.train import make_model

        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = dict(checkpoint["config"], device=args.device, amp=use_amp)
        if config["regime"] == "frozen" or config["dataset"] != args.dataset:
            parser.error("Checkpoint must contain an image encoder for the selected dataset.")

        model, _, transform = make_model(config, initialise_weights=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        encoder = model.encoder.eval()
        encoder_name = config["encoder"]

    elif args.encoder == "rn18":
        encoder = RN18Encoder(args.dataset, False, args.weights).to(device).eval()
        preprocessing = [transforms.ToTensor()]
        if args.dataset == "camelyon":
            preprocessing.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        transform = transforms.Compose(preprocessing)
        encoder_name = "rn18"

    else:
        if not args.weights:
            parser.error("--weights is required for foundation-model extraction.")
        encoder, transform, _ = load_foundation_encoder(args.encoder, args.weights, device)
        encoder_name = args.encoder

    # Store enough metadata to prevent accidentally mixing feature directories from different encoders.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "dataset": args.dataset,
        "encoder": encoder_name,
        "checkpoint": args.checkpoint,
        "weights": args.weights,
        "amp": use_amp,
        "augmentation": False,
    }
    settings_path = output_dir / "extraction.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings and not args.overwrite:
        raise ValueError("Output directory contains different extraction settings.")
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")

    # Extract every slide independently so one failure does not corrupt a multi-slide output file.
    manifest = read_manifest(args.csv, args.split)
    for row in manifest.itertuples():
        destination = output_dir / f"{row.slide_id}.pt"
        if destination.exists() and not args.overwrite:
            print(f"Exists: {destination}", flush=True)
            continue

        features = encode_tar(Path(args.tiles_dir) / f"{row.slide_id}.tar", encoder, transform,
                              args.batch_size, device, use_amp)
        features.update(label=int(row.label), slide_id=row.slide_id, encoder=encoder_name, augmentation=False)

        temporary = destination.with_suffix(".tmp")
        torch.save(features, temporary)
        temporary.replace(destination)
        print(f"{row.slide_id}: {tuple(features['features'].shape)}", flush=True)


if __name__ == "__main__":
    main()
