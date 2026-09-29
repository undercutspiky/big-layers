"""Foreground filtering for PANDA and TCGA-PRAD tile archives."""

import argparse
import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


BACKGROUND_THRESHOLD = 230
BLACK_THRESHOLD = 3
DEFAULT_TILE_SIZE = 256
DEFAULT_FOREGROUND_THRESHOLD = 0.60
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def foreground_fraction(image):
    """Return the fraction of pixels satisfying the PANDA/TCGA foreground rule."""
    gray = np.asarray(image.convert("L"))
    foreground = (BLACK_THRESHOLD < gray) & (gray < BACKGROUND_THRESHOLD)
    return float(np.count_nonzero(foreground)) / foreground.size


def filter_tar_file(src_tar_path, dst_tar_path, tile_size, foreground_threshold):
    """Copy only sufficiently foreground-rich image members into a new tar archive."""
    kept = 0
    considered = 0

    tmp_path = dst_tar_path.with_suffix(".tar.tmp")
    try:
        with tarfile.open(src_tar_path, "r") as tar_in, tarfile.open(tmp_path, "w") as tar_out:
            for member in tar_in.getmembers():
                if not member.isfile() or Path(member.name).suffix.lower() not in IMAGE_SUFFIXES:
                    continue

                considered += 1
                file_obj = tar_in.extractfile(member)
                if file_obj is None:
                    continue
                data = file_obj.read()

                image = Image.open(BytesIO(data)).convert("RGB")
                if image.size != (tile_size, tile_size):
                    print(
                        f"Skipping {src_tar_path.name}:{member.name}: "
                        f"got {image.size}, expected {(tile_size, tile_size)}.",
                        flush=True,
                    )
                    continue

                if foreground_fraction(image) < foreground_threshold:
                    continue

                info = tarfile.TarInfo(name=member.name)
                info.size = len(data)
                info.mtime = member.mtime
                info.mode = member.mode
                tar_out.addfile(info, BytesIO(data))
                kept += 1

        tmp_path.replace(dst_tar_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return kept, considered


def main():
    """Filter PANDA/TCGA per-WSI tile tars using the foreground rule from the experiments."""
    parser = argparse.ArgumentParser(
        description=(
            "Keep 256x256 tiles whose grayscale foreground fraction is at least 0.60, "
            "with foreground defined as 3 < gray < 230."
        )
    )
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--dst-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--foreground-threshold", type=float, default=DEFAULT_FOREGROUND_THRESHOLD)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_dir = args.src_dir.resolve()
    dst_dir = args.dst_dir.resolve()
    if src_dir == dst_dir:
        raise ValueError("--src-dir and --dst-dir must be different directories.")

    tar_paths = sorted(src_dir.glob("*.tar"))
    if not tar_paths:
        raise FileNotFoundError(f"No .tar files found in {src_dir}.")

    dst_dir.mkdir(parents=True, exist_ok=True)
    total_kept = 0
    total_considered = 0

    for src_path in tar_paths:
        dst_path = dst_dir / src_path.name
        if dst_path.exists() and not args.overwrite:
            print(f"Skipping {src_path.name}: destination exists.", flush=True)
            continue

        kept, considered = filter_tar_file(
            src_path,
            dst_path,
            args.tile_size,
            args.foreground_threshold,
        )
        total_kept += kept
        total_considered += considered
        print(f"{src_path.name}: kept {kept}/{considered} tiles.", flush=True)

    print(f"Done: kept {total_kept}/{total_considered} processed tiles.", flush=True)


if __name__ == "__main__":
    main()
