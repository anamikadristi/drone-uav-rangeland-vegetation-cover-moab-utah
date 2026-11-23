"""Shared paths and small helpers for the Moab rangeland drone project.

Author: Anamika Dristi, Ph.D.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Large raw downloads go here. The folder is ignored by git.
# Set DRONE_DATA_DIR to keep the raw files somewhere else.
RAW = Path(os.environ.get("DRONE_DATA_DIR", REPO / "data" / "raw"))
WORK = RAW / "work"            # intermediate rasters (also not in git)
DATA = REPO / "data"           # small inputs kept in the repository
FIG = REPO / "outputs" / "figures"
TAB = REPO / "outputs" / "tables"
MAPS = REPO / "outputs" / "maps"   # small GeoTIFF results
for _p in (RAW, WORK, DATA, FIG, TAB, MAPS):
    _p.mkdir(parents=True, exist_ok=True)

USER_AGENT = "python-requests (public data download script)"
SEED = 20230505

# USGS data release https://doi.org/10.5066/P9MMCJET, site 7 (grazed rangeland)
MS_ORTHO = RAW / "2023-05-05_Moab_Site7_Multi_Ortho_5band.tif"
MS_DSM = RAW / "2023-05-05_Moab_Site7_Multi_DSM.tif"
RGB_ORTHO = RAW / "2023-05-05_Moab_Site7_RGB_Ortho_3band.tif"
RGB_DSM = RAW / "2023-05-05_Moab_Site7_RGB_DSM.tif"
LIDAR_LAS = RAW / "2023-05-05_Moab_Site7_Lidar_PointCloud.las"

# Class codes used in every script
CLASSES = {1: "Green vegetation", 2: "Dry vegetation / litter",
           3: "Bare ground", 4: "Shadow / other"}
# Colorblind-safe (Okabe-Ito) colors for the four classes
CLASS_COLORS = {1: "#009E73", 2: "#E69F00", 3: "#D9C8A9", 4: "#3B3B3B"}


def setup_plot():
    """Common matplotlib settings for all figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "figure.dpi": 100,
                         "savefig.dpi": 170, "savefig.bbox": "tight"})
    return plt


def ms_grid():
    """Transform, CRS and shape of the multispectral orthomosaic grid."""
    import rasterio
    with rasterio.open(MS_ORTHO) as src:
        return src.transform, src.crs, (src.height, src.width)


def extent_local(transform, shape):
    """Image extent in metres from the south-west corner, for map axes."""
    h, w = shape
    return [0, w * transform.a, 0, h * abs(transform.e)]


def block_mean(arr, n):
    """Mean of non-overlapping n x n pixel blocks (NaN aware); edges are trimmed."""
    import numpy as np
    h, w = (arr.shape[0] // n) * n, (arr.shape[1] // n) * n
    a = arr[:h, :w].reshape(h // n, n, w // n, n)
    return np.nanmean(a, axis=(1, 3))


def save_fig(fig, path, dpi=170):
    """Save a PNG and keep it under about 1.4 MB.

    Photo-like maps make large PNG files. If the file is too big I convert it
    to a 256-color palette image, and if needed lower the resolution to 150 dpi.
    """
    from PIL import Image
    for d in (dpi, 150):
        fig.savefig(path, dpi=d)
        if path.stat().st_size <= 1.4e6:
            return
        Image.open(path).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
        if path.stat().st_size <= 1.4e6:
            return
