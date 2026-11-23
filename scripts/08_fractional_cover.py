"""Step 8. Fractional cover maps, site-level cover with uncertainty, field check.

- Aggregate both class maps to 1 m, 5 m and 10 m cells: percent green
  vegetation, percent dry vegetation or litter, percent bare ground.
- Site-level cover with three kinds of uncertainty:
    (a) spatial: block bootstrap over the 10 m cells (how much the number
        would change if a different set of 10 m cells had been flown),
    (b) model: 30 random forests refit on resampled training blocks (step 6),
    (c) method: thresholds versus random forest.
- Field check: the BLM AIM plot USGS_NEON2 lies inside the mosaic. The crew
  read 150 line-point intercept points on three 25 m transects on 30 April
  2023, five days before the flight. I compare its bare soil and its
  plant-plus-litter cover with the drone cover inside a 30 m radius circle.
- Bare ground pattern: size of connected bare patches and where they are,
  with the slope from step 4.
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from pyproj import Transformer
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from matplotlib.colors import ListedColormap
from common import (save_fig, WORK, DATA, TAB, FIG, MAPS, SEED, CLASSES, CLASS_COLORS, setup_plot,
                    ms_grid, extent_local)

plt = setup_plot()
tr, crs, shape = ms_grid()
px = tr.a
cls = {}
for name, f in (("Index thresholds", "class_threshold.tif"), ("Random forest", "class_random_forest.tif")):
    with rasterio.open(MAPS / f) as src:
        cls[name] = src.read(1)


def fractions(arr, cell_m):
    """Percent of each class in square cells of cell_m metres (masked pixels left out)."""
    n = int(round(cell_m / px))
    h, w = (shape[0] // n) * n, (shape[1] // n) * n
    a = arr[:h, :w].reshape(h // n, n, w // n, n)
    valid = (a > 0).sum(axis=(1, 3)).astype(float)
    out = {}
    for k in CLASSES:
        with np.errstate(invalid="ignore", divide="ignore"):
            out[k] = np.where(valid > 0.5 * n * n, 100 * (a == k).sum(axis=(1, 3)) / valid, np.nan)
    return out, n


# ---- 1. fractional cover at three cell sizes -------------------------------------------------
frac = {}
rows = []
for name, arr in cls.items():
    for cell in (1, 5, 10):
        f, n = fractions(arr, cell)
        frac[(name, cell)] = f
        for k in CLASSES:
            v = f[k][np.isfinite(f[k])]
            rows.append({"method": name, "cell_m": cell, "class": CLASSES[k], "n_cells": v.size,
                         "mean_pct": v.mean(), "sd_pct": v.std(ddof=1), "min_pct": v.min(),
                         "p05_pct": np.percentile(v, 5), "median_pct": np.median(v),
                         "p95_pct": np.percentile(v, 95), "max_pct": v.max(),
                         "cv": v.std(ddof=1) / v.mean() if v.mean() > 0 else np.nan})
        if cell == 1:   # save the 1 m fractional cover as a small GeoTIFF
            tr1 = from_origin(tr.c, tr.f, n * px, n * px)
            stack = np.stack([f[1], f[2], f[3]]).astype("float32")
            tag = "threshold" if name.startswith("Index") else "random_forest"
            with rasterio.open(MAPS / f"fractional_cover_1m_{tag}.tif", "w", driver="GTiff",
                               height=stack.shape[1], width=stack.shape[2], count=3, dtype="float32",
                               crs=crs, transform=tr1, nodata=np.nan, compress="deflate") as dst:
                dst.write(stack)
                dst.descriptions = ("green vegetation %", "dry vegetation or litter %", "bare ground %")
scale_tab = pd.DataFrame(rows).round(3)
scale_tab.to_csv(TAB / "20_fractional_cover_by_cell_size.csv", index=False)
print(scale_tab[scale_tab["class"] != "Shadow / other"].to_string())

# ---- 2. site-level cover with uncertainty ------------------------------------------------------
rng = np.random.default_rng(SEED)
model_boot = pd.read_csv(TAB / "19_site_cover_model_bootstrap.csv")
site = []
for name, arr in cls.items():
    nvalid = (arr > 0).sum()
    f10, n10 = fractions(arr, 10)
    a = arr[:(shape[0] // n10) * n10, :(shape[1] // n10) * n10].reshape(shape[0] // n10, n10, shape[1] // n10, n10)
    valid10 = (a > 0).sum(axis=(1, 3)).ravel().astype(float)
    for k in CLASSES:
        cnt = (a == k).sum(axis=(1, 3)).ravel().astype(float)
        est = 100 * (arr == k).sum() / nvalid
        bs = []
        for _ in range(5000):
            i = rng.integers(0, cnt.size, cnt.size)
            bs.append(100 * cnt[i].sum() / valid10[i].sum())
        row = {"method": name, "class": CLASSES[k], "site_cover_pct": est,
               "spatial_ci_low": np.percentile(bs, 2.5), "spatial_ci_high": np.percentile(bs, 97.5)}
        if name == "Random forest":
            mb = model_boot[f"cover_pct_{k}"]
            row.update({"model_refit_min": mb.min(), "model_refit_max": mb.max(), "model_refit_sd": mb.std(ddof=1)})
        site.append(row)
site = pd.DataFrame(site).round(2)
site.to_csv(TAB / "21_site_cover.csv", index=False)
print(site.to_string())

# ---- 3. field check against the BLM AIM plot ----------------------------------------------------
aim = pd.read_csv(DATA / "blm_aim_plot_USGS_NEON2.csv", index_col=0)["value"]
lon, lat = float(aim["Longitude_NAD83"]), float(aim["Latitude_NAD83"])
ex, ny_ = Transformer.from_crs(4269, crs, always_xy=True).transform(lon, lat)
c0, r0 = ~tr * (ex, ny_)
rr, cc = np.ogrid[0:shape[0], 0:shape[1]]
circle = (rr - r0) ** 2 + (cc - c0) ** 2 <= (30.0 / px) ** 2
aim_bare = float(aim["BareSoilCover"])
aim_fol = float(aim["TotalFoliarCover"])
aim_lit = float(aim["FH_TotalLitterCover"])
aim_rock = float(aim["FH_RockCover"])
N_LPI = 150


def wilson(p, n, z=1.96):
    p = p / 100
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (mid - half), 100 * (mid + half)


chk = []
for name, arr in cls.items():
    a = arr[circle]
    a = a[a > 0]
    green, dry, bare, shadow = [100 * (a == k).mean() for k in CLASSES]
    for label, drone, field in (("Bare ground", bare, aim_bare),
                                ("Plant canopy + litter (+ rock in field data)", green + dry, aim_fol + aim_lit + aim_rock)):
        lo, hi = wilson(field, N_LPI)
        chk.append({"quantity": label, "method": name, "drone_pct_in_30m_circle": drone,
                    "field_aim_pct": field, "field_95ci_low": lo, "field_95ci_high": hi,
                    "difference_drone_minus_field": drone - field,
                    "inside_field_ci": bool(lo <= drone <= hi)})
    chk.append({"quantity": "Green vegetation (no field equivalent)", "method": name,
                "drone_pct_in_30m_circle": green})
    chk.append({"quantity": "Dry vegetation / litter (no direct field equivalent)", "method": name,
                "drone_pct_in_30m_circle": dry})
chk = pd.DataFrame(chk).round(2)
chk.to_csv(TAB / "22_field_check_blm_aim.csv", index=False)
print(chk.to_string())
print("AIM plot center in mosaic (m from SW corner):", round(c0 * px, 1), round((shape[0] - r0) * px, 1),
      " circle inside mosaic share:", round(circle.sum() * px * px / (np.pi * 900), 3))

# ---- 4. bare ground patches (random forest map) ----------------------------------------------------
rf = cls["Random forest"]
hz = np.load(WORK / "height_10cm.npz")
ny10, nx10 = hz["mask"].shape
tr10 = from_origin(tr.c, tr.f, 0.10, 0.10)            # same 10 cm grid as step 4


def share10(k):
    """Share of class k in each 10 cm cell (masked pixels count as not k)."""
    out = np.zeros((ny10, nx10), "float32")
    reproject((rf == k).astype("float32"), out, src_transform=tr, src_crs=crs,
              dst_transform=tr10, dst_crs=crs, resampling=Resampling.average)
    return out


share = {k: share10(k) for k in (1, 2, 3)}
bare10 = share[3]
solid = bare10 >= 0.9                          # cells that are almost all bare
lab, nl = ndi.label(solid)
sizes = ndi.sum(solid, lab, np.arange(1, nl + 1)) * 0.01          # m2
tot = sizes.sum()
patch = pd.DataFrame([{
    "cells_10cm_at_least_90pct_bare_share_pct": 100 * solid.mean(),
    "n_patches": int(nl), "largest_patch_m2": sizes.max(), "median_patch_m2": np.median(sizes),
    "share_of_solid_bare_area_in_patches_over_0.25m2_pct": 100 * sizes[sizes > 0.25].sum() / tot,
    "share_of_solid_bare_area_in_patches_over_1m2_pct": 100 * sizes[sizes > 1].sum() / tot,
    "share_of_solid_bare_area_in_patches_over_5m2_pct": 100 * sizes[sizes > 5].sum() / tot,
    "site_area_in_patches_over_1m2_pct": 100 * sizes[sizes > 1].sum() / (solid.size * 0.01)}]).round(2)
patch.T.to_csv(TAB / "23_bare_patch_summary.csv", header=["value"])
print(patch.T.to_string())
big = np.isin(lab, np.nonzero(sizes > 1)[0] + 1)

# slope against bare fraction at 5 m cells
terr = np.load(WORK / "terrain_10cm.npz")
sl = terr["slope_pct"]
m5 = 50
hs, ws = (sl.shape[0] // m5) * m5, (sl.shape[1] // m5) * m5
slope5 = sl[:hs, :ws].reshape(hs // m5, m5, ws // m5, m5).mean(axis=(1, 3))
elev5 = terr["ground"][:hs, :ws].reshape(hs // m5, m5, ws // m5, m5).mean(axis=(1, 3))
b5 = frac[("Random forest", 5)][3]
g5 = frac[("Random forest", 5)][1]
mm = min(slope5.shape[0], b5.shape[0]), min(slope5.shape[1], b5.shape[1])
ok = np.isfinite(b5[:mm[0], :mm[1]])
rel = pd.DataFrame([
    {"pair": "bare % vs slope % (5 m cells)", "n": int(ok.sum()),
     "spearman_r": spearmanr(b5[:mm[0], :mm[1]][ok], slope5[:mm[0], :mm[1]][ok])[0],
     "p_value": spearmanr(b5[:mm[0], :mm[1]][ok], slope5[:mm[0], :mm[1]][ok])[1]},
    {"pair": "bare % vs elevation (5 m cells)", "n": int(ok.sum()),
     "spearman_r": spearmanr(b5[:mm[0], :mm[1]][ok], elev5[:mm[0], :mm[1]][ok])[0],
     "p_value": spearmanr(b5[:mm[0], :mm[1]][ok], elev5[:mm[0], :mm[1]][ok])[1]},
    {"pair": "green % vs elevation (5 m cells)", "n": int(ok.sum()),
     "spearman_r": spearmanr(g5[:mm[0], :mm[1]][ok], elev5[:mm[0], :mm[1]][ok])[0],
     "p_value": spearmanr(g5[:mm[0], :mm[1]][ok], elev5[:mm[0], :mm[1]][ok])[1]}]).round(4)
rel.to_csv(TAB / "24_bare_ground_vs_terrain.csv", index=False)
print(rel.to_string())

# height by class (10 cm cells where one class covers at least 90%)
hrows = []
for k in (1, 2, 3):
    sel = (share[k] >= 0.9) & hz["mask"]
    for src_name in ("SfM_multispectral", "SfM_natural_color", "Lidar"):
        v = hz[src_name][sel].astype("float64") * 100
        hrows.append({"class": CLASSES[k], "height_source": src_name, "n_cells": int(sel.sum()),
                      "mean_height_cm": v.mean(), "median_height_cm": np.median(v),
                      "p90_height_cm": np.percentile(v, 90),
                      "share_above_10cm_pct": 100 * (v > 10).mean()})
hcls = pd.DataFrame(hrows).round(2)
hcls.to_csv(TAB / "25_height_by_class.csv", index=False)
print(hcls.to_string())

# ---- 5. figures -------------------------------------------------------------------------------------
ext = extent_local(tr, shape)
cmap = ListedColormap(["#FFFFFF"] + [CLASS_COLORS[k] for k in CLASSES])
rgb = np.load(WORK / "rgb_on_ms.npy")
fig, ax = plt.subplots(2, 3, figsize=(15, 9.6))
st = 4
ax[0, 0].imshow(np.clip(np.moveaxis(rgb[:, ::st, ::st], 0, -1) / 190, 0, 1), extent=ext)
ax[0, 0].set_title("Natural color")
for a_, name in zip(ax[0, 1:], cls):
    a_.imshow(cls[name][::st, ::st], cmap=cmap, vmin=0, vmax=4, extent=ext, interpolation="nearest")
    a_.set_title(f"Class map: {name}")
r1, c1, nz = 2600, 2700, 500
ez = [0, nz * px, 0, nz * px]
ax[1, 0].imshow(np.clip(np.moveaxis(rgb[:, r1:r1 + nz, c1:c1 + nz], 0, -1) / 190, 0, 1), extent=ez)
ax[1, 0].set_title("Natural color, 6 m window")
for a_, name in zip(ax[1, 1:], cls):
    a_.imshow(cls[name][r1:r1 + nz, c1:c1 + nz], cmap=cmap, vmin=0, vmax=4, extent=ez, interpolation="nearest")
    a_.set_title(f"{name}, 6 m window")
for a_ in ax[0]:
    a_.set_xlabel("Easting from SW corner (m)"); a_.set_ylabel("Northing from SW corner (m)")
    a_.add_patch(plt.Rectangle((c1 * px, (shape[0] - r1 - nz) * px), nz * px, nz * px, fill=False, ec="k", lw=1.2))
for a_ in ax[1]:
    a_.set_xlabel("m"); a_.set_ylabel("m")
handles = [plt.Rectangle((0, 0), 1, 1, fc=CLASS_COLORS[k], ec="k") for k in CLASSES]
fig.legend(handles, list(CLASSES.values()), loc="lower center", ncol=4, frameon=False)
fig.suptitle("Classified maps, Moab site 7 (EPSG:6341). The black box is the 6 m window.")
fig.tight_layout(rect=[0, 0.03, 1, 1])
save_fig(fig, FIG / "10_class_maps.png")
plt.close(fig)

fig, ax = plt.subplots(3, 3, figsize=(13, 12))
for i, cell in enumerate((1, 5, 10)):
    f = frac[("Random forest", cell)]
    for j, (k, cm_, vmax) in enumerate(((1, "Greens", 40), (2, "YlOrBr", 60), (3, "copper_r", 100))):
        im = ax[i, j].imshow(f[k], extent=[0, f[k].shape[1] * cell, shape[0] * px - f[k].shape[0] * cell, shape[0] * px],
                             cmap=cm_, vmin=0, vmax=vmax)
        ax[i, j].set_title(f"{CLASSES[k]}, {cell} m cells")
        ax[i, j].set_xlabel("Easting from SW corner (m)"); ax[i, j].set_ylabel("Northing from SW corner (m)")
        plt.colorbar(im, ax=ax[i, j], shrink=0.8, label="cover (%)")
        if cell == 10:
            for (yy, xx), v in np.ndenumerate(f[k]):
                if np.isfinite(v):
                    ax[i, j].text(xx * 10 + 5, shape[0] * px - yy * 10 - 5, f"{v:.0f}", ha="center", va="center",
                                  fontsize=7, color="white" if v > 0.55 * vmax else "black")
fig.suptitle("Fractional cover from the random forest map at 1 m, 5 m and 10 m cells (EPSG:6341)")
fig.tight_layout()
save_fig(fig, FIG / "11_fractional_cover_maps.png")
plt.close(fig)

# bare ground pattern with terrain
fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
f1 = frac[("Random forest", 1)][3]
e1 = [0, f1.shape[1], shape[0] * px - f1.shape[0], shape[0] * px]
im = ax[0].imshow(f1, extent=e1, cmap="copper_r", vmin=20, vmax=100)
g = terr["ground"]
xs = np.linspace(0, g.shape[1] * 0.1, g.shape[1]); ys = np.linspace(g.shape[0] * 0.1, 0, g.shape[0])
cs = ax[0].contour(xs, ys, g, levels=np.arange(np.floor(g.min() * 5) / 5, g.max(), 0.2), colors="#0072B2", linewidths=0.8)
ax[0].clabel(cs, fmt="%.1f", fontsize=7)
st_ = 60
ax[0].quiver(xs[st_ // 2::st_], ys[st_ // 2::st_], -terr["gx"][st_ // 2::st_, st_ // 2::st_],
             terr["gy"][st_ // 2::st_, st_ // 2::st_], color="#0072B2", scale=0.6, width=0.004)
ax[0].set_title("(a) Bare ground (%), 1 m cells, with 0.2 m contours\nand downslope arrows")
plt.colorbar(im, ax=ax[0], shrink=0.8, label="bare ground (%)")
show = np.where(big, 2, np.where(solid, 1, 0))
ax[1].imshow(show, extent=[0, show.shape[1] * 0.1, shape[0] * px - show.shape[0] * 0.1, shape[0] * px],
             cmap=ListedColormap(["#F2F2F2", "#E6B88A", "#8C2D04"]), interpolation="nearest")
ax[1].set_title("(b) Connected bare patches (10 cm cells at least 90% bare)\ndark = patches larger than 1 m2")
order = np.sort(sizes)[::-1]
ax[2].plot(order, 100 * np.cumsum(order) / tot, color="#8C2D04")
ax[2].set_xscale("log"); ax[2].invert_xaxis()
ax[2].set_xlabel("Patch size (m2), largest first"); ax[2].set_ylabel("Cumulative share of solid bare area (%)")
ax[2].set_title("(c) Bare patch size distribution"); ax[2].grid(alpha=0.3)
for a_ in ax[:2]:
    a_.set_xlabel("Easting from SW corner (m)"); a_.set_ylabel("Northing from SW corner (m)")
fig.tight_layout()
save_fig(fig, FIG / "12_bare_ground_pattern.png")
plt.close(fig)

# field check and site cover bars
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
x = np.arange(3)
for i, (name, col) in enumerate((("Index thresholds", "#E69F00"), ("Random forest", "#0072B2"))):
    d = site[(site.method == name) & (site["class"] != "Shadow / other")]
    v = d.site_cover_pct.values
    err = np.vstack([v - d.spatial_ci_low.values, d.spatial_ci_high.values - v])
    ax[0].bar(x + (i - 0.5) * 0.36, v, 0.34, yerr=err, color=col, capsize=3, label=name)
    for xi, vi in zip(x + (i - 0.5) * 0.36, v):
        ax[0].text(xi, vi + 2.5, f"{vi:.1f}", ha="center", fontsize=8)
ax[0].set_xticks(x); ax[0].set_xticklabels(["Green vegetation", "Dry vegetation / litter", "Bare ground"])
ax[0].set_ylabel("Site cover (%)"); ax[0].legend(); ax[0].grid(axis="y", alpha=0.3)
ax[0].set_title("(a) Site-level cover, whole mosaic\n(bars: 95% spatial block bootstrap)")
d = chk.dropna(subset=["field_aim_pct"])
labs = ["Bare ground", "Plant canopy + litter"]
for i, (name, col) in enumerate((("Index thresholds", "#E69F00"), ("Random forest", "#0072B2"))):
    v = d[d.method == name].drone_pct_in_30m_circle.values
    ax[1].bar(np.arange(2) + (i - 1) * 0.27, v, 0.25, color=col, label=f"Drone, {name.lower()}")
    for xi, vi in zip(np.arange(2) + (i - 1) * 0.27, v):
        ax[1].text(xi, vi + 1.5, f"{vi:.1f}", ha="center", fontsize=8)
fd = d[d.method == "Random forest"]
fv = fd.field_aim_pct.values
ax[1].bar(np.arange(2) + 0.27, fv, 0.25, color="#009E73", label="Field (BLM AIM, 150 points)",
          yerr=np.vstack([fv - fd.field_95ci_low.values, fd.field_95ci_high.values - fv]), capsize=3)
for xi, vi in zip(np.arange(2) + 0.27, fv):
    ax[1].text(xi + 0.06, vi + 1.5, f"{vi:.1f}", ha="left", fontsize=8)
ax[1].set_xticks(range(2)); ax[1].set_xticklabels(labs)
ax[1].set_ylabel("Cover in 30 m radius plot (%)"); ax[1].legend(fontsize=8); ax[1].grid(axis="y", alpha=0.3)
ax[1].set_ylim(0, 100)
ax[1].set_title("(b) Field check, plot USGS_NEON2\n(field bars: 95% binomial interval)")
fig.tight_layout()
save_fig(fig, FIG / "13_site_cover_and_field_check.png")
plt.close(fig)
print("step 8 done")
