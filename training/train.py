"""Train the paper configurations for PANDA/TCGA-PRAD and CAMELYON17/16."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from MIL.abmil import ABMIL, PandaRN18ABMIL
from MIL.dtfd import DTFDMIL
from MIL.transmil import TransMIL
from big_layers import cuda_config as cc
from training.amp import autocast_context, make_grad_scaler
from training.data import SlideBags, collate_bags, feature_path, load_features, read_manifest
from training.demon_adamw import DemonAdamW
from training.encoders import CAM_MEAN, CAM_STD, FM_NAMES, IMAGENET_MEAN, IMAGENET_STD
from training.encoders import H0MiniEncoder, RN18Encoder, load_foundation_encoder
from training.mil_wrapper import MILWrapper
from training.mil_wrapper_vit import MILWrapperViT
from training.transforms import FixedBatchAugment, FixedTrivialAugmentWide, TrainTransforms


# -----------------------------------------------------------------------------
# Model construction
# -----------------------------------------------------------------------------


def make_head(config, dimension):
    """Build the MIL head used for either image training or precomputed features."""
    hidden = 256 if config["regime"] == "h0" else 128
    classes = 6 if config["dataset"] == "panda" else 2

    if config["method"] == "abmil":
        head_cls = PandaRN18ABMIL if config["dataset"] == "panda" and config["encoder"] == "rn18" else ABMIL
        return head_cls(dimension, hidden, classes)

    if config["method"] == "dtfd":
        split_mode = "chunk" if config["dataset"] == "panda" and config["regime"] == "rn18" else "tensor_split"
        return DTFDMIL(dimension, hidden, classes, m_dim=512, num_groups=4, split_mode=split_mode)

    return TransMIL(dimension, hidden, classes)


def make_model(config, initialise_weights=True):
    """Build the encoder, MIL head, and train/evaluation transforms for one training regime."""
    dataset = config["dataset"]
    regime = config["regime"]
    device = config["device"]
    train_transform = None
    eval_transform = None

    if regime == "rn18":
        weights = config.get("weights") if initialise_weights else None
        encoder = RN18Encoder(dataset, not config["frozen"], weights, pretrained=initialise_weights,
                              num_big_stages=config["big_stages"], conv_elements=config["conv_elements"],
                              bn_elements=config["bn_elements"])
        dimension = encoder.feature_dim

        base_transforms = [transforms.ToTensor()]
        if dataset == "camelyon":
            normalization = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD) if config["frozen"] else \
                transforms.Normalize(CAM_MEAN, CAM_STD)
            base_transforms.append(normalization)

        eval_transform = transforms.Compose(base_transforms)
        train_transform = TrainTransforms(FixedBatchAugment(), base_transforms) if config["augmentation"] else \
            eval_transform

    elif regime == "h0":
        if initialise_weights:
            encoder, eval_transform, (mean, std) = load_foundation_encoder("h0-mini", config["weights"], device)
            config_path = Path(config["weights"]).with_name("config.json")
            config["h0_config"] = json.loads(config_path.read_text())
        else:
            import timm

            published = config["h0_config"]
            architecture = published.get("architecture", "vit_base_patch14_reg4_dinov2")
            backbone = timm.create_model(architecture, pretrained=False, mlp_layer=timm.layers.SwiGLUPacked,
                                         act_layer=nn.SiLU, **published.get("model_args", {}))
            encoder = H0MiniEncoder(backbone)
            mean = published["pretrained_cfg"]["mean"]
            std = published["pretrained_cfg"]["std"]
            eval_transform = transforms.Compose([
                transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        train_transform = FixedTrivialAugmentWide(mean=mean, std=std)
        dimension = 1536

    else:
        encoder = nn.Identity()
        dimension = config["feature_dim"]

    head = make_head(config, dimension)
    if regime == "h0":
        model = MILWrapperViT(encoder, head, config["micro_batch_size"], config["inference_batch_size"],
                              config["amp"], getattr(torch, config["amp_dtype"]))
    else:
        model = MILWrapper(encoder, head, config["frozen"], config["inference_batch_size"])

    return model.to(device), train_transform, eval_transform


# -----------------------------------------------------------------------------
# Reproducibility and evaluation
# -----------------------------------------------------------------------------


def seed_everything(seed):
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id):
    """Give each DataLoader worker deterministic Python/NumPy RNGs derived from PyTorch."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def metrics_from_predictions(labels, probabilities):
    """Compute slide-level metrics from probabilities over the complete evaluation cohort."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    prediction = probabilities.argmax(axis=1)
    result = {
        "accuracy": float(accuracy_score(labels, prediction)),
        "confusion_matrix": confusion_matrix(
            labels, prediction, labels=np.arange(probabilities.shape[1])
        ).tolist(),
    }

    if probabilities.shape[1] == 6:
        result["qwk"] = float(cohen_kappa_score(labels, prediction, weights="quadratic", labels=list(range(6))))
    else:
        result["auc"] = float(roc_auc_score(labels, probabilities[:, 1])) if len(np.unique(labels)) == 2 else None

    return result


@torch.no_grad()
def evaluate_model(model, loader, config):
    """Evaluate one trained model once and return metrics plus per-slide predictions."""
    model.eval()
    ids = []
    labels = []
    probabilities = []
    losses = []
    dtype = getattr(torch, config["amp_dtype"])
    device_type = torch.device(config["device"]).type

    for batch in loader:
        with autocast_context(config["amp"], dtype, device_type):
            output = model(batch["inputs"])
            logits = output[0]
            loss = nn.functional.cross_entropy(logits.float(), batch["label"].to(config["device"]))

        ids.extend(batch["slide_id"])
        labels.extend(batch["label"].tolist())
        probabilities.extend(logits.float().softmax(-1).cpu().tolist())
        losses.append(float(loss))

    result = metrics_from_predictions(labels, probabilities)
    result["loss"] = float(np.mean(losses))
    return result, ids, labels, probabilities


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


def train_epoch(model, loader, criterion, optimizer, scaler, config, epoch):
    """Train one epoch with gradient accumulation and optional H0-mini LwF regularisation."""
    model.train()

    # H0-mini starts with one frozen-encoder warm-up epoch, then trains the full encoder.
    warmup = config["regime"] == "h0" and epoch < config["warmup_epochs"]
    if config["regime"] == "h0":
        model.encoder.requires_grad_(not warmup)
        model.encoder.train(not warmup)

    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    dtype = getattr(torch, config["amp_dtype"])
    device_type = torch.device(config["device"]).type

    for step, batch in enumerate(loader, 1):
        labels = batch["label"].to(config["device"])

        # Forward pass and complete slide-level loss.
        with autocast_context(config["amp"], dtype, device_type):
            output = model(batch["inputs"])
            logits, auxiliary = output[:2]
            loss = criterion(logits, labels)

            if config["method"] == "dtfd":
                loss = loss + sum(criterion(prediction, labels) for prediction in auxiliary)

            if config["regime"] == "h0":
                teacher = batch["teacher"].to(config["device"]).reshape(-1, output[2].shape[-1])
                loss = loss + config["lwf_weight"] * nn.functional.mse_loss(output[2].float(), teacher.float())

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}, batch {step}.")

        total_loss += float(loss.detach())
        divisor = config["accumulation"] if config["regime"] == "h0" else 1
        scaler.scale(loss / divisor).backward()

        # The ViT wrapper now has dL/dE and can replay encoder micro-batches to recover dL/dtheta.
        if config["regime"] == "h0":
            model.backward_encoder()

        # Update after the requested number of slide batches.
        if step % config["accumulation"] == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    return total_loss / len(loader)


# -----------------------------------------------------------------------------
# Command-line configuration
# -----------------------------------------------------------------------------


def build_parser():
    """Build one CLI for both datasets and all main-paper training regimes."""
    parser = argparse.ArgumentParser(description=__doc__)

    data = parser.add_argument_group("data")
    data.add_argument("--dataset", choices=["panda", "camelyon"], required=True)
    data.add_argument("--train-csv", required=True, help="Training manifest or combined split manifest.")
    data.add_argument("--val-csv", help="Validation manifest; omit only if --train-csv has a split column.")
    data.add_argument("--data-dir", required=True, help="Tile tar files, or .pt files for frozen-feature training.")
    data.add_argument("--output-dir", required=True)
    data.add_argument("--workers", type=int, default=4)

    model = parser.add_argument_group("model")
    model.add_argument("--regime", choices=["rn18", "h0", "frozen"], required=True)
    model.add_argument("--method", choices=["abmil", "dtfd", "transmil"], required=True)
    model.add_argument("--encoder", choices=["rn18", *FM_NAMES], help="Encoder used for precomputed frozen features.")
    model.add_argument("--weights", help="Local ImageNet or foundation-model checkpoint.")
    model.add_argument("--frozen", action="store_true", help="Keep RN18 frozen while training the MIL head.")
    model.add_argument("--augmentation", action="store_true", help="Use the WSI-level augmentation condition.")
    model.add_argument("--inference-batch-size", type=int, default=32)

    optimisation = parser.add_argument_group("optimisation")
    optimisation.add_argument("--epochs", type=int)
    optimisation.add_argument("--lr", type=float)
    optimisation.add_argument("--batch-size", type=int)
    optimisation.add_argument("--bag-size", type=int)
    optimisation.add_argument("--accumulation", type=int)
    optimisation.add_argument("--seed", type=int, default=42)
    optimisation.add_argument("--device", default="cuda:0")
    optimisation.add_argument("--no-amp", action="store_true")
    optimisation.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    optimisation.add_argument("--resume", help="Resume from a last.pt written by this training script.")

    big_layers = parser.add_argument_group("RN18 Big Layers")
    big_layers.add_argument("--big-stages", type=int, choices=range(5), default=0)
    big_layers.add_argument("--conv-elements", type=int, default=60_000_000)
    big_layers.add_argument("--bn-elements", type=int, default=6_000_000)

    h0 = parser.add_argument_group("H0-mini")
    h0.add_argument("--micro-batch-size", type=int, help="Encoder images per recomputed micro-batch.")
    h0.add_argument("--teacher-dir", help="Unaugmented H0-mini embeddings for LwF.")

    return parser


def resolve_config(args, parser):
    """Apply the paper defaults and reject combinations that do not belong to the selected regime."""
    config = vars(args).copy()
    config["val_csv"] = args.val_csv or args.train_csv
    config["amp"] = not args.no_amp and torch.device(args.device).type == "cuda" and args.regime != "frozen"

    if args.regime == "rn18":
        config["encoder"] = "rn18"
    elif args.regime == "h0":
        config["encoder"] = "h0-mini"
        config["frozen"] = False
        config["augmentation"] = True
    else:
        config["encoder"] = args.encoder or "rn18"
        config["frozen"] = True
        config["augmentation"] = False

    if args.regime == "h0" and (args.micro_batch_size is None or args.teacher_dir is None):
        parser.error("H0-mini training requires --micro-batch-size and --teacher-dir.")
    if args.regime == "h0" and not args.weights and not args.resume:
        parser.error("A new H0-mini run requires --weights.")
    if args.regime != "h0" and (args.micro_batch_size is not None or args.teacher_dir is not None):
        parser.error("--micro-batch-size and --teacher-dir are only used for H0-mini training.")
    if args.regime != "rn18" and (args.big_stages != 0 or args.conv_elements != 60_000_000 or
                                  args.bn_elements != 6_000_000):
        parser.error("Big-Layer stage/chunk arguments are only used for RN18 training.")

    defaults = {
        "epochs": 20 if args.dataset == "panda" or args.regime == "h0" else 120 if config["frozen"] else 90,
        "lr": 5e-5 if args.regime == "h0" or (args.dataset == "camelyon" and not config["frozen"]) else 1e-4,
        "batch_size": 1 if args.regime in ("h0", "frozen") else 2,
        "bag_size": 256 if args.dataset == "panda" else 1024 if config["frozen"] else 512,
        "accumulation": (32 if args.dataset == "panda" else 2) if args.regime == "h0" else
                        1 if args.regime == "frozen" else 16 if args.dataset == "panda" else 2,
    }
    for key, value in defaults.items():
        if config[key] is None:
            config[key] = value

    config["warmup_epochs"] = 1 if args.regime == "h0" else 0
    config["lwf_weight"] = (1.0 if args.dataset == "panda" else 10.0) if args.regime == "h0" else 0.0
    config["sort_tiles"] = args.dataset == "camelyon" and args.regime != "h0" and (
        (args.regime == "frozen" and config["encoder"] != "rn18") or
        (args.regime == "rn18" and not config["frozen"] and args.method != "abmil")
    )

    positive = ("epochs", "lr", "batch_size", "bag_size", "accumulation", "inference_batch_size")
    if any(config[key] <= 0 for key in positive):
        parser.error("Epochs, learning rate, and batch sizes must be positive.")
    if args.workers < 0:
        parser.error("--workers must be nonnegative.")

    if not args.val_csv:
        import pandas as pd

        if "split" not in pd.read_csv(args.train_csv, nrows=1):
            parser.error("Supply --val-csv when --train-csv has no split column.")

    return config


# -----------------------------------------------------------------------------
# Training setup and execution
# -----------------------------------------------------------------------------


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = resolve_config(args, parser)

    # Output directory and optional epoch-boundary resume.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "last.pt").exists() and not args.resume:
        parser.error("Output already contains last.pt. Use --resume or choose another output directory.")

    cc.configure(args.device, config["amp"])
    seed_everything(args.seed)
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None

    if checkpoint:
        old = checkpoint["config"]
        keys = (
            "dataset", "regime", "method", "encoder", "frozen", "epochs", "lr", "bag_size", "batch_size",
            "accumulation", "seed", "augmentation", "big_stages", "micro_batch_size", "amp", "amp_dtype"
        )
        for key in keys:
            if config.get(key) != old.get(key):
                parser.error(f"Resume configuration differs for {key}: {config.get(key)} versus {old.get(key)}.")
        config.update({key: old[key] for key in ("h0_config", "feature_dim") if key in old})

    # Determine feature dimensionality before constructing a frozen-feature MIL head.
    if args.regime == "frozen":
        first_slide = read_manifest(args.train_csv, "train")["slide_id"].iloc[0]
        config["feature_dim"] = load_features(feature_path(args.data_dir, first_slide))[0].shape[1]

    model, train_transform, val_transform = make_model(config, initialise_weights=checkpoint is None)

    # Datasets and DataLoaders.
    train_data = SlideBags(args.train_csv, args.data_dir, args.dataset, "train", config["bag_size"], True,
                           train_transform, args.regime == "frozen", config.get("teacher_dir"),
                           args.regime == "rn18" and args.dataset == "panda", config["sort_tiles"])
    val_bag_size = 2048 if args.dataset == "camelyon" else None
    val_data = SlideBags(config["val_csv"], args.data_dir, args.dataset, "val", val_bag_size, False,
                         val_transform, args.regime == "frozen", sort_tiles=config["sort_tiles"])

    overlap = set(train_data.df["slide_id"]) & set(val_data.df["slide_id"])
    if overlap:
        raise ValueError(f"Train/validation overlap: {len(overlap)} slides.")

    loader_kwargs = {
        "num_workers": args.workers,
        "pin_memory": config["amp"],
        "collate_fn": collate_bags,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, **loader_kwargs)

    # Loss, optimiser, scheduler, and mixed precision.
    criterion = nn.CrossEntropyLoss(weight=train_data.class_weights().to(args.device))
    if args.dataset == "camelyon" and args.regime == "rn18" and not config["frozen"]:
        optimizer = DemonAdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4,
                               total_steps=config["epochs"] * len(train_loader))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)

    if args.regime == "h0":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.955)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    scaler = make_grad_scaler(config["amp"] and args.amp_dtype == "float16")

    # Restore optimiser state only after all objects have been created.
    start_epoch = 0
    best_score = -float("inf")
    best_loss = float("inf")
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        best_loss = checkpoint["best_loss"]

    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Training slides: {len(train_data)}; validation slides: {len(val_data)}", flush=True)
    print(json.dumps(config, indent=2), flush=True)

    # Main training loop. Validation selects the best checkpoint; external test cohorts are never used here.
    for epoch in range(start_epoch, config["epochs"]):
        seed_everything(args.seed + epoch)
        started = time.perf_counter()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, config, epoch)
        scheduler.step()
        validation, _, _, _ = evaluate_model(model, val_loader, config)

        metric = "qwk" if args.dataset == "panda" else "accuracy"
        score = validation[metric]
        improved = score > best_score or (score == best_score and validation["loss"] < best_loss)
        if improved:
            best_score = score
            best_loss = validation["loss"]

        saved = {
            "config": config,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": best_score,
            "best_loss": best_loss,
        }
        torch.save(saved, output_dir / "last.tmp")
        (output_dir / "last.tmp").replace(output_dir / "last.pt")
        if improved:
            torch.save(saved, output_dir / "best.tmp")
            (output_dir / "best.tmp").replace(output_dir / "best.pt")

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation": validation,
            "seconds": time.perf_counter() - started,
            "lr": optimizer.param_groups[0]["lr"],
        }
        with (output_dir / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
