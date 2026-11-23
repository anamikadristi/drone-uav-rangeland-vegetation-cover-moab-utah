"""Step 4. Vegetation height and micro-relief from the surface models.

The data release has three independent height sources for the same plot:
  - the DSM from the multispectral flight (structure from motion, 2.4 cm cells),
  - the DSM from the natural color flight (structure from motion, 1.5 cm cells),
  - a drone lidar point cloud (YellowScan Vx20, about 500 points per m2).
Plants here are short (the field crew measured about 10 cm for grasses and
about 30 cm for shrubs), so this is a hard test for height mapping.

Method: I estimate the ground under each cell with a morphological opening
(a moving minimum followed by a moving maximum, 1 m window) and a light
smoothing. Height above ground = surface - ground. I do this on a common
10 cm grid for the three sources, and at full resolution for the
multispectral DSM so that height can be used as a classifier feature.
I also compute slope from a 1 m smoothed DSM for the runoff discussion.
"""
import numpy as np
import pandas as pd
import laspy
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from common import (save_fig, WORK, FIG, TAB, MS_DSM, RGB_DSM, LIDAR_LAS, setup_plot,
                    ms_grid, extent_local)

plt = setup_plot()
tr, crs, shape = ms_grid()
mask = np.load(WORK / "valid_mask.npy")


def height_above_ground(dsm, cell, window_m=1.0):
    """Surface minus a morphological-opening ground estimate (metres)."""
    k = max(3, int(round(window_m / cell)) | 1)
    ground = ndi.grey_opening(dsm, size=(k, k))
    ground = ndi.gaussian_filter(ground, sigma=k / 4)
    return np.clip(dsm - ground, 0, None), ground


# ---- 1. full resolution height from the multispectral DSM (classifier feature) --
with rasterio.open(MS_DSM) as src:
    dsm_ms = src.read(1)
    cell_ms = src.res[0]
h_ms, ground_ms = height_above_ground(dsm_ms, cell_ms)
h_on_ms = np.zeros(shape, "float32")
with rasterio.open(MS_DSM) as src:
    reproject(h_ms, h_on_ms, src_transform=src.transform, src_crs=src.crs,
              dst_transform=tr, dst_crs=crs, resampling=Resampling.bilinear)
np.save(WORK / "height_ms_dsm.npy", h_on_ms)

# ---- 2. common 10 cm grid for the three sources ---------------------------------
cell = 0.10
x0, y1 = tr.c, tr.f
nx, ny = int(shape[1] * tr.a / cell), int(shape[0] * tr.a / cell)
tr10 = from_origin(x0, y1, cell, cell)


def to_grid(path, how):
    out = np.zeros((ny, nx), "float32")
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, 1), out, dst_transform=tr10, dst_crs=crs, resampling=how)
    return out


surf = {"SfM multispectral": to_grid(MS_DSM, Resampling.max),
        "SfM natural color": to_grid(RGB_DSM, Resampling.max)}

las = laspy.read(LIDAR_LAS)
lx, ly, lz = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
ci = ((lx - x0) / cell).astype(int)
ri = ((y1 - ly) / cell).astype(int)
ok = (ci >= 0) & (ci < nx) & (ri >= 0) & (ri < ny)
flat = ri[ok] * nx + ci[ok]
zmax = np.full(ny * nx, -np.inf, "float64")
np.maximum.at(zmax, flat, lz[ok])
count = np.bincount(flat, minlength=ny * nx).reshape(ny, nx)
zmax = zmax.reshape(ny, nx)
empty = ~np.isfinite(zmax)
# fill the few empty cells with the nearest filled cell
nearest = ndi.distance_transform_edt(empty, return_distances=False, return_indices=True)
zmax = zmax[tuple(nearest)].astype("float32")
surf["Lidar (max per cell)"] = zmax
lidar_class = np.unique(np.asarray(las.classification), return_counts=True)

heights, grounds = {}, {}
for k, v in surf.items():
    heights[k], grounds[k] = height_above_ground(v, cell)

mask10 = np.zeros((ny, nx), "float32")
reproject(mask.astype("float32"), mask10, src_transform=tr, src_crs=crs,
          dst_transform=tr10, dst_crs=crs, resampling=Resampling.average)
m10 = mask10 > 0.99
np.savez(WORK / "height_10cm.npz", **{k.split(" (")[0].replace(" ", "_"): v for k, v in heights.items()},
         mask=m10)

# ---- 3. tables ---------------------------------------------------------------------
rows = []
for k, v in heights.items():
    x = v[m10]
    rows.append({"source": k, "cell_m": cell,
                 "mean_height_cm": 100 * x.mean(), "median_height_cm": 100 * np.median(x),
                 "p95_height_cm": 100 * np.percentile(x, 95), "p99_height_cm": 100 * np.percentile(x, 99),
                 "max_height_cm": 100 * x.max(),
                 "area_above_10cm_pct": 100 * (x > 0.10).mean(),
                 "area_above_20cm_pct": 100 * (x > 0.20).mean()})
ht = pd.DataFrame(rows).round(2)
ht.to_csv(TAB / "07_height_summary.csv", index=False)
print(ht.to_string())

names = list(heights)
rows = []
for i in range(3):
    for j in range(i + 1, 3):
        a, b = heights[names[i]][m10], heights[names[j]][m10]
        za, zb = surf[names[i]][m10], surf[names[j]][m10]
        both = ((a > 0.10) & (b > 0.10)).sum() / max(((a > 0.10) | (b > 0.10)).sum(), 1)
        rows.append({"source_a": names[i], "source_b": names[j],
                     "pearson_r_height": np.corrcoef(a, b)[0, 1],
                     "spearman_r_height": spearmanr(a[::7], b[::7])[0],
                     "overlap_of_cells_above_10cm_iou": both,
                     "mean_elevation_difference_cm": 100 * (za - zb).mean(),
                     "sd_elevation_difference_cm": 100 * (za - zb).std()})
cmp_ = pd.DataFrame(rows).round(3)
cmp_.to_csv(TAB / "08_height_source_agreement.csv", index=False)
print(cmp_.to_string())
print("lidar classes (code, count):", lidar_class, " mean points per 10 cm cell:", count[m10].mean().round(2),
      " empty cells %:", round(100 * empty[m10].mean(), 2))
pd.DataFrame({"lidar_points_total": [len(lx)],
              "lidar_points_per_m2": [round(len(lx) / (nx * ny * cell * cell), 1)],
              "lidar_ground_class_share_pct": [round(100 * (np.asarray(las.classification) == 2).mean(), 2)],
              "empty_10cm_cells_pct": [round(100 * empty[m10].mean(), 2)]}).to_csv(
    TAB / "09_lidar_point_summary.csv", index=False)

# ---- 4. slope from a smoothed 1 m surface -------------------------------------------
g = ndi.gaussian_filter(grounds["SfM natural color"], sigma=1.0 / cell)     # 1 m smoothing
gy, gx = np.gradient(g, cell)
slope_pct = 100 * np.hypot(gx, gy)
aspect_down = (np.degrees(np.arctan2(-gx, gy)) + 360) % 360     # compass direction of steepest descent
np.savez(WORK / "terrain_10cm.npz", ground=g, slope_pct=slope_pct, gx=gx, gy=gy)
terr = pd.DataFrame([{"relief_m": float(g[m10].max() - g[m10].min()),
                      "min_elev_m_navd88": float(g[m10].min()), "max_elev_m_navd88": float(g[m10].max()),
                      "mean_slope_pct": float(slope_pct[m10].mean()),
                      "p95_slope_pct": float(np.percentile(slope_pct[m10], 95)),
                      "mean_downslope_direction_deg": float((np.degrees(np.arctan2(-gx[m10].mean(), gy[m10].mean())) + 360) % 360)}]).round(3)
terr.to_csv(TAB / "10_terrain_summary.csv", index=False)
print(terr.to_string())

# ---- 5. figure -----------------------------------------------------------------------
ext = [0, nx * cell, 0, ny * cell]
fig, ax = plt.subplots(2, 2, figsize=(11.5, 10))
im = ax[0, 0].imshow(g, extent=ext, cmap="cividis")
cs = ax[0, 0].contour(np.linspace(0, nx * cell, nx), np.linspace(ny * cell, 0, ny), g,
                      levels=np.arange(np.floor(g.min() * 5) / 5, g.max(), 0.2), colors="k", linewidths=0.5)
ax[0, 0].clabel(cs, fmt="%.1f", fontsize=7)
ax[0, 0].set_title("(a) Ground elevation, 1 m smoothing (m, NAVD88)\ncontours every 0.2 m")
plt.colorbar(im, ax=ax[0, 0], shrink=0.8, label="elevation (m)")
for a, k, t in zip([ax[0, 1], ax[1, 0], ax[1, 1]], names, ["(b)", "(c)", "(d)"]):
    im = a.imshow(np.where(m10, heights[k], np.nan) * 100, extent=ext, cmap="viridis", vmin=0, vmax=25)
    a.set_title(f"{t} Height above ground: {k}")
    plt.colorbar(im, ax=a, shrink=0.8, label="height (cm)")
for a in ax.ravel():
    a.set_xlabel("Easting from SW corner (m)")
    a.set_ylabel("Northing from SW corner (m)")
fig.suptitle("Terrain and height above ground on a 10 cm grid (EPSG:6341)")
fig.tight_layout()
save_fig(fig, FIG / "06_height_and_terrain.png")
plt.close(fig)
print("step 4 done")
