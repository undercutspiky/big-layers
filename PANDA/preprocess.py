"""OpenSlide tiling for PANDA; see README.md for the complete workflow."""
from preprocessing.tile_wsi import main


if __name__ == "__main__":
    main(default_tissue_filter="none")
