"""OpenSlide WSI tiling used by the PANDA/TCGA and CAMELYON preprocessing workflows."""

import argparse
import io
import math
import tarfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu


DEFAULT_TILE_SIZE = 256
DEFAULT_MAGNIFICATION = 20.0
DEFAULT_EDGE_MIN_REMAINDER = 50
DEFAULT_MASK_LEVEL = 5
DEFAULT_TISSUE_THRESHOLD = 0.8
DEFAULT_JPEG_QUALITY = 95


def get_base_magnification(slide, override=None):
    """Return the scanner objective magnification used at OpenSlide level 0."""
    if override is not None:
        return float(override)

    for key in (
        "openslide.objective-power",
        "aperio.AppMag",
        "hamamatsu.SourceLens",
    ):
        value = slide.properties.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except ValueError:
            continue

    raise RuntimeError(
        "Could not determine the slide objective magnification. "
        "Pass --base-magnification explicitly for this scanner/file format."
    )


def choose_read_level(slide, target_downsample):
    """Choose the pyramid level closest to the requested target downsample."""
    level = slide.get_best_level_for_downsample(float(target_downsample))
    return int(level), float(slide.level_downsamples[level])


def axis_origins(length, step, edge_min_remainder):
    """Return grid origins plus the historical overlapping edge-aligned origin."""
    if length < step:
        return []

    n_full = length // step
    origins = [(index, index * step) for index in range(n_full)]
    remainder = length % step

    # Historical preprocessing intentionally adds a right/bottom aligned tile when
    # enough pixels remain. This overlaps the previous grid tile by construction.
    if remainder > edge_min_remainder:
        origins.append((n_full, length - step))

    return origins


def create_otsu_hsv_mask(image, close_kernel_size=50, open_kernel_size=30):
    """Create the HSV/Otsu tissue mask used for CAMELYON preprocessing."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    h_threshold = threshold_otsu(h)
    s_threshold = threshold_otsu(s)
    v_threshold = threshold_otsu(v)

    lower = np.array([h_threshold, s_threshold, 70], dtype=np.uint8)
    upper = np.array([180, 255, v_threshold], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel_close = np.ones((close_kernel_size, close_kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    kernel_open = np.ones((open_kernel_size, open_kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    return mask > 0


def load_mask(slide, requested_level):
    """Read one low-resolution OpenSlide level and build its Otsu tissue mask."""
    mask_level = min(int(requested_level), slide.level_count - 1)
    width, height = slide.level_dimensions[mask_level]
    image = slide.read_region((0, 0), mask_level, (width, height)).convert("RGB")
    mask = create_otsu_hsv_mask(np.asarray(image))
    return mask, float(slide.level_downsamples[mask_level]), mask_level


def tissue_fraction(mask, mask_downsample, x0, y0, footprint):
    """Return Otsu-mask coverage for one tile footprint specified in level-0 pixels."""
    x_start = max(0, int(math.floor(x0 / mask_downsample)))
    y_start = max(0, int(math.floor(y0 / mask_downsample)))
    x_end = min(mask.shape[1], int(math.ceil((x0 + footprint) / mask_downsample)))
    y_end = min(mask.shape[0], int(math.ceil((y0 + footprint) / mask_downsample)))

    region = mask[y_start:y_end, x_start:x_end]
    if region.size == 0:
        return 0.0
    return float(region.mean())


def read_tile(slide, x0, y0, level, level_downsample, footprint, tile_size):
    """Read one target-magnification tile using OpenSlide only."""
    read_size = max(1, int(round(footprint / level_downsample)))
    tile = slide.read_region((int(x0), int(y0)), level, (read_size, read_size)).convert("RGB")
    if tile.size != (tile_size, tile_size):
        tile = tile.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
    return tile


def save_tile(tile, archive, row, col, quality):
    """JPEG-encode one tile and append it to a per-WSI tar archive."""
    buffer = io.BytesIO()
    tile.save(buffer, format="JPEG", quality=quality)
    data = buffer.getvalue()

    info = tarfile.TarInfo(name=f"tile_{row}_{col}.jpg")
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


def process_slide(slide_path, output_dir, target_magnification, tile_size, edge_min_remainder,
                  tissue_filter, mask_level, min_tissue_fraction, jpeg_quality,
                  base_magnification=None, overwrite=False):
    """Tile one WSI into a tar archive, optionally filtering with a CAMELYON Otsu mask."""
    output_path = Path(output_dir) / f"{Path(slide_path).stem}.tar"
    if output_path.exists() and not overwrite:
        print(f"Skipping {Path(slide_path).name}: {output_path.name} already exists.", flush=True)
        return

    import openslide
    slide = openslide.OpenSlide(str(slide_path))
    try:
        base_mag = get_base_magnification(slide, base_magnification)
        if base_mag < target_magnification:
            raise ValueError(
                f"{slide_path}: base magnification {base_mag:g}x is below requested "
                f"{target_magnification:g}x."
            )

        target_downsample = base_mag / target_magnification
        footprint = max(1, int(round(tile_size * target_downsample)))
        level, level_downsample = choose_read_level(slide, target_downsample)
        width0, height0 = slide.dimensions

        rows = axis_origins(height0, footprint, edge_min_remainder)
        cols = axis_origins(width0, footprint, edge_min_remainder)
        if not rows or not cols:
            print(f"Skipping {Path(slide_path).name}: slide is smaller than one tile.", flush=True)
            return

        mask = None
        mask_downsample = None
        used_mask_level = None
        if tissue_filter == "otsu":
            mask, mask_downsample, used_mask_level = load_mask(slide, mask_level)

        print(
            f"Processing {Path(slide_path).name}: L0={width0}x{height0}, "
            f"base={base_mag:g}x, target={target_magnification:g}x, "
            f"read_level={level}, footprint={footprint}px"
            + (f", mask_level={used_mask_level}" if used_mask_level is not None else ""),
            flush=True,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(".tar.tmp")
        kept = 0
        considered = 0

        try:
            with tarfile.open(tmp_path, "w") as archive:
                for row, y0 in rows:
                    for col, x0 in cols:
                        considered += 1
                        if mask is not None:
                            fraction = tissue_fraction(mask, mask_downsample, x0, y0, footprint)
                            if fraction < min_tissue_fraction:
                                continue

                        tile = read_tile(slide, x0, y0, level, level_downsample, footprint, tile_size)
                        save_tile(tile, archive, row, col, jpeg_quality)
                        kept += 1

            tmp_path.replace(output_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    finally:
        slide.close()

    print(f"Saved {output_path}: kept {kept}/{considered} tiles.", flush=True)


def collect_slides(input_dir, extensions):
    """Recursively collect WSIs matching the requested file extensions."""
    suffixes = {f".{extension.lower().lstrip('.')}" for extension in extensions}
    return sorted(
        path for path in Path(input_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def main(default_tissue_filter="none"):
    """Tile PANDA/TCGA WSIs or tile-and-filter CAMELYON WSIs."""
    parser = argparse.ArgumentParser(
        description="Create one tar archive of 20x 256x256 JPEG tiles per WSI using OpenSlide."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extensions", nargs="+", default=["svs", "tif", "tiff"])
    parser.add_argument("--target-magnification", type=float, default=DEFAULT_MAGNIFICATION)
    parser.add_argument("--base-magnification", type=float, default=None)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--edge-min-remainder", type=int, default=DEFAULT_EDGE_MIN_REMAINDER)
    parser.add_argument("--tissue-filter", choices=["none", "otsu"], default=default_tissue_filter)
    parser.add_argument("--mask-level", type=int, default=DEFAULT_MASK_LEVEL)
    parser.add_argument("--min-tissue-fraction", type=float, default=DEFAULT_TISSUE_THRESHOLD)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--exclude", nargs="*", default=[], help="Slide stems to skip.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    slides = collect_slides(args.input_dir, args.extensions)
    excluded = set(args.exclude)
    slides = [slide for slide in slides if slide.stem not in excluded]
    if not slides:
        raise FileNotFoundError(
            f"No WSIs with extensions {args.extensions} were found under {args.input_dir}."
        )

    print(f"Found {len(slides)} slides.", flush=True)
    for slide_path in slides:
        process_slide(
            slide_path=slide_path,
            output_dir=args.output_dir,
            target_magnification=args.target_magnification,
            tile_size=args.tile_size,
            edge_min_remainder=args.edge_min_remainder,
            tissue_filter=args.tissue_filter,
            mask_level=args.mask_level,
            min_tissue_fraction=args.min_tissue_fraction,
            jpeg_quality=args.jpeg_quality,
            base_magnification=args.base_magnification,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
