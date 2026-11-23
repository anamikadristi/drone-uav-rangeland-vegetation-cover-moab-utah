"""Step 3. Vegetation indices from the multispectral orthomosaic.

Near infrared indices (need the red edge and NIR bands):
  NDVI  = (NIR - red) / (NIR + red)
  NDRE  = (NIR - red edge) / (NIR + red edge)
  SAVI  = 1.5 (NIR - red) / (NIR + red + 0.5)
  MSAVI = (2 NIR + 1 - sqrt((2 NIR + 1)^2 - 8 (NIR - red))) / 2
Visible indices (what an RGB-only drone could give). They are computed from
the blue, green and red reflectance bands, with r, g, b = band / (R + G + B):
  ExG  = 2g - r - b
  ExGR = ExG - (1.4 r - g)
  VARI = (G - R) / (G + R - B)
  GLI  = (2G - R - B) / (2G + R + B)
I also keep a soil redness index, (red - blue) / (red + blue), because the
soil at this site is red and plants are not.
"""
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from common import save_fig, WORK, FIG, TAB, setup_plot, ms_grid, extent_local

plt = setup_plot()
tr, crs, shape = ms_grid()
ms = np.load(WORK / "ms_corrected.npy")
mask = np.load(WORK / "valid_mask.npy")
B, G, R, RE, N = ms
eps = 1e-6

idx = {}
idx["NDVI"] = (N - R) / (N + R + eps)
idx["NDRE"] = (N - RE) / (N + RE + eps)
idx["SAVI"] = 1.5 * (N - R) / (N + R + 0.5)
idx["MSAVI"] = (2 * N + 1 - np.sqrt(np.maximum((2 * N + 1) ** 2 - 8 * (N - R), 0))) / 2
s = R + G + B + eps
r, g, b = R / s, G / s, B / s
idx["ExG"] = 2 * g - r - b
idx["ExGR"] = idx["ExG"] - (1.4 * r - g)
vden = G + R - B
idx["VARI"] = np.clip((G - R) / np.where(np.abs(vden) < eps, eps, vden), -1, 1)
idx["GLI"] = (2 * G - R - B) / (2 * G + R + B + eps)
idx["REDNESS"] = (R - B) / (R + B + eps)
del r, g, b, s, vden

np.savez(WORK / "indices.npz", **{k: v.astype("float32") for k, v in idx.items()})

# ---- table of percentiles and Otsu thresholds --------------------------------
q = [1, 5, 25, 50, 75, 95, 99]
rows = []
sub = (slice(None, None, 4), slice(None, None, 4))
m = mask[sub]
for k, v in idx.items():
    x = v[sub][m]
    p = np.percentile(x, q)
    rows.append({"index": k, **{f"p{a}": b for a, b in zip(q, p)},
                 "mean": x.mean(), "otsu_threshold": threshold_otsu(x)})
tab = pd.DataFrame(rows).round(4)
tab.to_csv(TAB / "05_index_percentiles.csv", index=False)
print(tab.to_string())

# rank correlation between indices (how much the RGB indices agree with NDVI)
samp = np.stack([v[sub][m][::50] for v in idx.values()])
rho = pd.DataFrame(np.corrcoef(np.argsort(np.argsort(samp, axis=1), axis=1)),
                   index=idx.keys(), columns=idx.keys()).round(3)
rho.to_csv(TAB / "06_index_rank_correlation.csv")
print(rho["NDVI"])

# ---- histograms ------------------------------------------------------------------
fig, ax = plt.subplots(3, 3, figsize=(12, 9))
for a, (k, v) in zip(ax.ravel(), idx.items()):
    x = v[sub][m]
    lo, hi = np.percentile(x, [0.2, 99.8])
    a.hist(x, bins=120, range=(lo, hi), color="#0072B2", alpha=0.85)
    t = tab.set_index("index").loc[k, "otsu_threshold"]
    a.axvline(t, color="#D55E00", lw=1.5, label=f"Otsu = {t:.3f}")
    a.set_yscale("log")
    a.set_title(k)
    a.set_xlabel("index value (-)")
    a.set_ylabel("pixels (log scale)")
    a.legend(fontsize=8)
fig.suptitle("Index histograms, Moab site 7 (every 4th pixel, artifacts masked)")
fig.tight_layout()
save_fig(fig, FIG / "03_index_histograms.png")
plt.close(fig)

# ---- maps ------------------------------------------------------------------------
ext = extent_local(tr, shape)
st = 4
show = ["NDVI", "NDRE", "MSAVI", "ExG", "VARI", "REDNESS"]
fig, ax = plt.subplots(2, 3, figsize=(14, 9))
for a, k in zip(ax.ravel(), show):
    v = np.where(mask, idx[k], np.nan)[::st, ::st]
    lo, hi = np.nanpercentile(v, [2, 99.5])
    im = a.imshow(v, extent=ext, cmap="viridis", vmin=lo, vmax=hi)
    a.set_title(k)
    a.set_xlabel("Easting from SW corner (m)")
    a.set_ylabel("Northing from SW corner (m)")
    plt.colorbar(im, ax=a, shrink=0.75, label=f"{k} (-)")
fig.suptitle("Index maps, Moab site 7, 5 May 2023 (EPSG:6341)")
fig.tight_layout()
save_fig(fig, FIG / "04_index_maps.png")
plt.close(fig)

# ---- zoom: 6 m x 6 m window so single plants can be seen ---------------------------
rgb = np.load(WORK / "rgb_on_ms.npy")
r0, c0, n = 2600, 2700, 500
fig, ax = plt.subplots(1, 4, figsize=(15, 4.3))
e = [0, n * tr.a, 0, n * tr.a]
ax[0].imshow(np.clip(np.moveaxis(rgb[:, r0:r0 + n, c0:c0 + n], 0, -1) / 190, 0, 1), extent=e)
ax[0].set_title("Natural color")
for a, k in zip(ax[1:], ["NDVI", "ExG", "REDNESS"]):
    v = idx[k][r0:r0 + n, c0:c0 + n]
    im = a.imshow(v, extent=e, cmap="viridis", vmin=np.percentile(v, 1), vmax=np.percentile(v, 99.7))
    a.set_title(k)
    plt.colorbar(im, ax=a, shrink=0.8)
for a in ax:
    a.set_xlabel("m")
    a.set_ylabel("m")
fig.suptitle("6 m by 6 m window near the plot center")
fig.tight_layout()
save_fig(fig, FIG / "05_index_zoom.png")
plt.close(fig)
print("step 3 done")
