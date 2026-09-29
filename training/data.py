"""Datasets and manifest helpers for tile archives and precomputed MIL features."""

import random
import tarfile
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


# -----------------------------------------------------------------------------
# Manifests and feature files
# -----------------------------------------------------------------------------


def read_manifest(path, split=None, num_classes=None):
    """Read one slide per row and normalize the PANDA/TCGA column names used by this repository."""
    df = pd.read_csv(path)

    # TCGA-PRAD metadata: convert primary/secondary Gleason patterns to ISUP grade groups.
    if {"scan_name_aperio", "gleason_1", "gleason_2"} <= set(df):
        gleason_1 = df["gleason_1"].astype(str).str.extract(r"([0-5])", expand=False).astype(int)
        gleason_2 = df["gleason_2"].astype(str).str.extract(r"([0-5])", expand=False).astype(int)
        total = gleason_1 + gleason_2
        df["label"] = np.select(
            [total <= 6, (gleason_1 == 3) & (gleason_2 == 4), (gleason_1 == 4) & (gleason_2 == 3),
             total == 8, total >= 9],
            [1, 2, 3, 4, 5],
            default=-1,
        )
        df = df.rename(columns={"scan_name_aperio": "slide_id"})
    else:
        df = df.rename(columns={"FILENAME": "slide_id", "isup_grade": "label"})

    if not {"slide_id", "label"} <= set(df):
        raise ValueError(f"{path}: expected slide_id,label or the documented PANDA/TCGA columns.")

    if split is not None and "split" in df:
        df = df[df["split"].astype(str).str.lower() == split.lower()].copy()
    if df.empty:
        raise ValueError(f"{path}: no slides for split {split!r}.")
    if df[["slide_id", "label"]].isna().any().any():
        raise ValueError(f"{path}: missing slide ID or label.")

    df["slide_id"] = df["slide_id"].astype(str).map(lambda value: value.removesuffix(".tar").removesuffix(".svs"))
    if df["slide_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate slide IDs.")
    if df["slide_id"].map(lambda value: "/" in value or "\\" in value or value in (".", "..")).any():
        raise ValueError("Slide IDs must be basenames, not paths.")

    labels = pd.to_numeric(df["label"], errors="raise")
    if (labels != labels.astype(int)).any() or (labels < 0).any():
        raise ValueError(f"{path}: labels must be nonnegative integer class indices.")
    df["label"] = labels.astype(int)

    if num_classes is not None and (df["label"] >= num_classes).any():
        raise ValueError(f"{path}: a label exceeds num_classes={num_classes}.")

    return df.reset_index(drop=True)


def feature_path(root, slide_id):
    """Return <slide_id>.pt, with support for the older <slide_id>_*_feats.pt naming convention."""
    path = Path(root) / f"{slide_id}.pt"
    if path.is_file():
        return path

    matches = sorted(Path(root).glob(f"{slide_id}_*_feats.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one feature file for {slide_id} in {root}; found {len(matches)}.")
    return matches[0]


def load_features(path):
    """Load one N x D feature matrix and the tile names that define its row order."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or "features" not in obj or "tile_names" not in obj:
        raise ValueError(f"{path}: expected a dictionary with features and tile_names.")

    features = obj["features"]
    names = list(obj["tile_names"])
    if features.ndim != 2 or len(features) == 0 or len(features) != len(names):
        raise ValueError(f"{path}: invalid N x D features or tile-name count.")
    if len(set(names)) != len(names) or not torch.isfinite(features).all():
        raise ValueError(f"{path}: duplicate tile names or non-finite features.")

    return features.float(), names


# -----------------------------------------------------------------------------
# Tar-file cache
# -----------------------------------------------------------------------------


class ArchiveCache:
    """Keep a small per-worker LRU of open tar files instead of one handle per slide."""

    def __init__(self, capacity=16):
        self.capacity = capacity
        self.handles = OrderedDict()

    def get(self, path):
        if path not in self.handles:
            if len(self.handles) >= self.capacity:
                self.handles.popitem(last=False)[1].close()
            self.handles[path] = tarfile.open(path, "r")
        self.handles.move_to_end(path)
        return self.handles[path]

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def __getstate__(self):
        return {"capacity": self.capacity, "handles": OrderedDict()}

    def __del__(self):
        self.close()


# -----------------------------------------------------------------------------
# WSI bags
# -----------------------------------------------------------------------------


class SlideBags(Dataset):
    """Load one WSI bag from a tar archive or a precomputed feature file.

    The paper configurations use different short-bag handling: PANDA RN18 pads with white tiles,
    PANDA H0-mini samples with replacement, and CAMELYON samples additional tiles with replacement.
    WSI-level augmentation parameters are generated once and reused for every tile in the bag.
    """

    def __init__(self, manifest, root, dataset, split=None, bag_size=None, training=False, transform=None,
                 features=False, teacher_root=None, white_padding=False, sort_tiles=False):
        self.df = read_manifest(manifest, split, 6 if dataset == "panda" else 2)
        self.root = Path(root)
        self.dataset = dataset
        self.bag_size = bag_size
        self.training = training
        self.transform = transform
        self.features = features
        self.teacher_root = Path(teacher_root) if teacher_root else None
        self.white_padding = white_padding
        self.sort_tiles = sort_tiles
        self.archives = ArchiveCache()
        self.members = {}

        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        if bag_size is not None and bag_size < 1:
            raise ValueError("bag_size must be positive or None for all tiles.")

        # Cache tile names once. Workers open tar files lazily through ArchiveCache.
        if not features:
            for slide_id in self.df["slide_id"]:
                path = self.root / f"{slide_id}.tar"
                with tarfile.open(path, "r") as archive:
                    names = [member.name for member in archive
                             if member.isfile() and Path(member.name).suffix.lower() in IMAGE_SUFFIXES]
                if not names or len(set(names)) != len(names):
                    raise ValueError(f"{path}: empty bag or duplicate image names.")
                self.members[slide_id] = names
        else:
            for slide_id in self.df["slide_id"]:
                feature_path(self.root, slide_id)

    def __len__(self):
        return len(self.df)

    def class_weights(self):
        """Return inverse-frequency class weights using the weighting convention used in the experiments."""
        counts = np.zeros(6 if self.dataset == "panda" else 2, dtype=np.float64)
        for row in self.df.itertuples():
            if self.dataset == "panda" and not self.features:
                counts[row.label] += len(self.members[row.slide_id])
            else:
                counts[row.label] += 1

        if (counts == 0).any():
            raise ValueError(f"Training split is missing a class: counts={counts.tolist()}.")
        weights = 1 / counts
        return torch.tensor(weights / weights.sum(), dtype=torch.float32)

    def _select_indices(self, names):
        """Choose the tile indices for one bag while preserving the dataset-specific sampling behaviour."""
        num_tiles = len(names)
        bag_size = self.bag_size
        indices = list(range(num_tiles))

        if bag_size is not None and bag_size < num_tiles:
            indices = random.sample(indices, bag_size)
        elif self.training and bag_size is not None and bag_size > num_tiles and not self.white_padding:
            if self.dataset == "panda":
                indices = random.choices(indices, k=bag_size)
            else:
                indices += random.choices(indices, k=bag_size - num_tiles)
                random.shuffle(indices)
        elif self.training or self.dataset == "camelyon":
            random.shuffle(indices)

        selected_names = [names[index] for index in indices]
        if self.sort_tiles:
            order = sorted(range(len(indices)), key=lambda index: selected_names[index])
            indices = [indices[index] for index in order]
            selected_names = [selected_names[index] for index in order]

        return indices, selected_names

    def _load_tiles(self, slide_id, selected_names):
        """Load and transform selected image tiles from a WSI tar archive."""
        archive = self.archives.get(self.root / f"{slide_id}.tar")
        params = self.transform.generate_params() if hasattr(self.transform, "generate_params") else None
        tiles = []

        for name in selected_names:
            with archive.extractfile(name) as stream:
                image = Image.open(BytesIO(stream.read())).convert("RGB")
            tiles.append(self.transform(image, params) if params is not None else self.transform(image))

        bag = torch.stack(tiles)
        if self.training and self.white_padding and self.bag_size is not None and len(bag) < self.bag_size:
            padding = torch.ones((self.bag_size - len(bag), *bag.shape[1:]), dtype=bag.dtype)
            bag = torch.cat((bag, padding))
        return bag

    def __getitem__(self, index):
        row = self.df.iloc[index]
        slide_id = row["slide_id"]
        label = int(row["label"])

        if self.features:
            values, names = load_features(feature_path(self.root, slide_id))
        else:
            values = None
            names = self.members[slide_id]

        indices, selected_names = self._select_indices(names)
        bag = values[indices] if self.features else self._load_tiles(slide_id, selected_names)
        sample = {"inputs": bag.contiguous(), "label": torch.tensor(label), "slide_id": slide_id}

        # LwF teacher embeddings are matched by tile name because sampled bags may contain duplicates.
        if self.teacher_root is not None:
            teacher, teacher_names = load_features(feature_path(self.teacher_root, slide_id))
            lookup = {name: index for index, name in enumerate(teacher_names)}
            missing = set(selected_names) - lookup.keys()
            if missing:
                example = next(iter(missing))
                raise KeyError(f"{slide_id}: {len(missing)} selected tiles have no teacher embedding (e.g. {example}).")
            sample["teacher"] = teacher[[lookup[name] for name in selected_names]]

        return sample


def collate_bags(samples):
    """Stack fixed-size bags; all-tile evaluation uses batch_size=1."""
    shapes = {tuple(sample["inputs"].shape) for sample in samples}
    if len(shapes) != 1:
        raise ValueError("Variable-length bags need batch_size=1 or a fixed training bag size.")

    result = {
        "inputs": torch.stack([sample["inputs"] for sample in samples]),
        "label": torch.stack([sample["label"] for sample in samples]),
        "slide_id": [sample["slide_id"] for sample in samples],
    }
    if "teacher" in samples[0]:
        result["teacher"] = torch.stack([sample["teacher"] for sample in samples])
    return result
