"""Evaluate trained checkpoints on all tiles or repeated limited-tile samples."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from big_layers import cuda_config as cc
from training.data import SlideBags, collate_bags
from training.mil_wrapper import MILWrapper
from training.train import evaluate_model, make_head, make_model, seed_everything, seed_worker


# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", required=True, help="Independent runs of one configuration.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"])

    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--tiles-dir")
    data.add_argument("--features-dir", help="Features must come from the same encoder as the checkpoint.")

    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tiles", type=int, default=0, help="0 = all tiles; paper subsets use 256 or 2048.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-amp", action="store_true")
    return parser


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.max_tiles < 0 or args.repeats < 1 or args.inference_batch_size < 1 or args.workers < 0:
        parser.error("Invalid tile count, repeats, workers, or inference batch size.")

    # Adapted encoders have checkpoint-specific features, so they cannot share one feature directory.
    if args.features_dir and len(args.checkpoint) > 1:
        regimes = [torch.load(path, map_location="cpu", weights_only=True)["config"]["regime"]
                   for path in args.checkpoint]
        if any(regime != "frozen" for regime in regimes):
            parser.error("Evaluate adapted encoders separately when using precomputed features.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_means = []
    repeat_records = []
    reference_profile = None

    for model_index, checkpoint_path in enumerate(args.checkpoint):
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = dict(saved["config"], device=args.device, inference_batch_size=args.inference_batch_size)
        profile = {key: config.get(key) for key in
                   ("dataset", "regime", "method", "encoder", "frozen", "augmentation")}

        if reference_profile is not None and profile != reference_profile:
            parser.error("Aggregate only independent runs from the same training condition.")
        reference_profile = profile

        config["amp"] = config["amp"] and not args.no_amp and torch.device(args.device).type == "cuda"
        cc.configure(args.device, config["amp"])
        use_features = args.features_dir is not None

        # Build either a feature-only MIL head or the complete image model.
        if config["regime"] == "frozen" and not use_features:
            parser.error("Frozen-feature checkpoints require --features-dir.")

        if use_features:
            dimension = config.get(
                "feature_dim",
                1536 if config["regime"] == "h0" else 32768 if config["dataset"] == "panda" else 512,
            )
            head = make_head(config, dimension)
            head_state = {key.removeprefix("mil."): value for key, value in saved["model"].items()
                          if key.startswith("mil.")}
            head.load_state_dict(head_state)
            model = MILWrapper(nn.Identity(), head, freeze_encoder=True).to(args.device)
            transform = None
        else:
            model, _, transform = make_model(config, initialise_weights=False)
            model.load_state_dict(saved["model"], strict=True)

        # Evaluation loader. Variable-length/all-tile bags use one slide per batch.
        data_root = args.features_dir or args.tiles_dir
        dataset = SlideBags(args.csv, data_root, config["dataset"], args.split, args.max_tiles or None, False,
                            transform, use_features, sort_tiles=config["sort_tiles"])
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                            collate_fn=collate_bags, worker_init_fn=seed_worker)

        # Repeated limited-tile evaluations are averaged within a trained model.
        results = []
        for repeat in range(args.repeats):
            seed_everything(args.seed + repeat)
            result, ids, labels, probabilities = evaluate_model(model, loader, config)
            result.update(model=str(checkpoint_path), repeat=repeat, slides=len(ids))
            results.append(result)

            frame = pd.DataFrame({
                "slide_id": ids,
                "label": labels,
                "prediction": np.argmax(probabilities, axis=1),
            })
            for class_index, values in enumerate(np.asarray(probabilities).T):
                frame[f"probability_{class_index}"] = values
            frame.to_csv(output_dir / f"model{model_index}_repeat{repeat:03d}.csv", index=False)

        metric_names = ("accuracy", "qwk") if config["dataset"] == "panda" else ("accuracy", "auc")
        means = {
            metric: float(np.mean([result[metric] for result in results]))
            if all(result[metric] is not None for result in results) else None
            for metric in metric_names
        }
        model_means.append(means)
        repeat_records.extend(results)

        del model, saved

    # Standard deviation is across independently trained models, not repeated tile samples.
    summary = {
        metric: {
            "mean": float(np.mean([result[metric] for result in model_means])),
            "sd": float(np.std([result[metric] for result in model_means], ddof=1)) if len(model_means) > 1 else None,
        }
        for metric in model_means[0]
        if all(result[metric] is not None for result in model_means)
    }

    report = {
        "protocol": vars(args),
        "model_means": model_means,
        "across_models": summary,
        "repeats": repeat_records,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
