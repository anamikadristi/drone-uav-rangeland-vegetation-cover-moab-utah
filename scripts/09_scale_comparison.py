"""Step 9. Scale comparison: drone cover against Sentinel-2 NDVI.

Sentinel-2 L2A scenes used (10 m pixels, surface reflectance, cloud free
over the site): 29 April 2023 (6 days before the flight) and 9 May 2023
(4 days after). The drone flight was on 5 May 2023.

For every Sentinel-2 pixel that lies fully inside the drone mosaic I average
the drone data inside that pixel footprint:
  - percent green vegetation, percent green + dry, percent bare (from step 6),
  - red and NIR reflectance, then NDVI from those means.
Then I compare with the Sentinel-2 NDVI of the same pixel: scatter plots,
Pearson and Spearman correlation, and an ordinary least squares line.

The site is only 72 m by 70 m, so there are about 40 pixels, and cover does
not vary much between them. I also check how the result changes if the
satellite image is shifted by up to 10 m, because Sentinel-2 geolocation is
only good to several metres.
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from scipy import stats
from common import save_fig, WORK, DATA, TAB, FIG, MAPS, setup_plot, ms_grid

plt = setup_plot()
tr, crs, shape = ms_grid()
mask = np.load(WORK / "valid_mask.npy")
ms = np.load(WORK / "ms_corrected.npy")
with rasterio.open(MAPS / "class_random_forest.tif") as src:
    rf = src.read(1)
with rasterio.open(MAPS / "class_threshold.tif") as src:
    th = src.read(1)

# ---- Sentinel-2 windows -------------------------------------------------------------
s2 = {}
for f in sorted((DATA / "sentinel2").glob("*_window.tif")):
    with rasterio.open(f) as src:
        a = src.read()
        s2_tr, s2_crs, s2_shape = src.transform, src.crs, (src.height, src.width)
        date = src.tags()["datetime"][:10]
    s2[date] = {"blue": a[0], "green": a[1], "red": a[2], "nir": a[3], "scl": a[4],
                "ndvi": (a[3] - a[2]) / (a[3] + a[2])}
dates = sorted(s2)
s2["mean of both"] = {k: (s2[dates[0]][k] + s2[dates[1]][k]) / 2 for k in ("red", "nir")}
s2["mean of both"]["ndvi"] = (s2[dates[0]]["ndvi"] + s2[dates[1]]["ndvi"]) / 2
print("Sentinel-2 dates:", dates, "CRS", s2_crs, "scene classes over window:",
      {d: np.unique(s2[d]["scl"]).tolist() for d in dates})

layers = {"valid": mask.astype("float32"),
          "green": ((rf == 1) & mask).astype("float32"),
          "dry": ((rf == 2) & mask).astype("float32"),
          "bare": ((rf == 3) & mask).astype("float32"),
          "green_thr": ((th == 1) & mask).astype("float32"),
          "red": np.where(mask, ms[2], 0).astype("float32"),
          "nir": np.where(mask, ms[4], 0).astype("float32")}


# First put every drone layer on a 0.5 m grid that nests inside the Sentinel-2
# grid (20 x 20 fine cells per satellite pixel). This is done once; the
# satellite-pixel means, also for shifted grids, are then simple block means.
FINE = 0.5
NSUB = int(10 / FINE)
fine_tr = Affine(FINE, 0, s2_tr.c, 0, -FINE, s2_tr.f)
fine_shape = (s2_shape[0] * NSUB, s2_shape[1] * NSUB)
fine = {}
for k, v in layers.items():
    dst = np.zeros(fine_shape, "float32")
    reproject(v, dst, src_transform=tr, src_crs=crs, dst_transform=fine_tr, dst_crs=s2_crs,
              resampling=Resampling.average)
    fine[k] = dst


def aggregate(dx=0.0, dy=0.0):
    """Drone layers averaged into Sentinel-2 pixels; the satellite grid is moved by (dx, dy) metres."""
    sx, sy = int(round(dx / FINE)), int(round(dy / FINE))
    out = {}
    for k, v in fine.items():
        # moving the satellite grid east by dx is the same as rolling the drone layer west
        vv = np.roll(v, (sy, -sx), axis=(0, 1))
        out[k] = vv.reshape(s2_shape[0], NSUB, s2_shape[1], NSUB).mean(axis=(1, 3))
    ok = out["valid"] > 0.98          # pixel fully inside the mosaic (artifacts are under 2%)
    res = {"ok": ok}
    for k in ("green", "dry", "bare", "green_thr"):
        res[k] = 100 * out[k] / out["valid"]
    res["red"] = out["red"] / out["valid"]
    res["nir"] = out["nir"] / out["valid"]
    res["ndvi"] = (res["nir"] - res["red"]) / (res["nir"] + res["red"])
    return res


with np.errstate(invalid="ignore", divide="ignore"):
    d0 = aggregate()
ok = d0["ok"]
n = int(ok.sum())
print("Sentinel-2 pixels fully inside the mosaic:", n)

rows_i, cols_i = np.nonzero(ok)
pix = pd.DataFrame({"s2_row": rows_i, "s2_col": cols_i,
                    "x_center_utm12n_wgs84": s2_tr.c + (cols_i + 0.5) * 10,
                    "y_center_utm12n_wgs84": s2_tr.f - (rows_i + 0.5) * 10,
                    "drone_green_pct": d0["green"][ok], "drone_dry_pct": d0["dry"][ok],
                    "drone_bare_pct": d0["bare"][ok], "drone_green_pct_threshold_map": d0["green_thr"][ok],
                    "drone_red": d0["red"][ok], "drone_nir": d0["nir"][ok], "drone_ndvi": d0["ndvi"][ok]})
for d in dates:
    tag = d.replace("-", "")
    pix[f"s2_red_{tag}"] = s2[d]["red"][ok]
    pix[f"s2_nir_{tag}"] = s2[d]["nir"][ok]
    pix[f"s2_ndvi_{tag}"] = s2[d]["ndvi"][ok]
pix["s2_ndvi_mean"] = s2["mean of both"]["ndvi"][ok]
pix.round(4).to_csv(TAB / "26_scale_comparison_pixels.csv", index=False)


def relation(x, y, xname, yname, date):
    lr = stats.linregress(x, y)
    tcrit = stats.t.ppf(0.975, len(x) - 2)
    sp = stats.spearmanr(x, y)
    return {"x": xname, "y": yname, "s2_date": date, "n": len(x), "pearson_r": lr.rvalue,
            "r2": lr.rvalue ** 2, "p_value": lr.pvalue, "spearman_r": sp[0], "spearman_p": sp[1],
            "slope": lr.slope, "slope_ci_low": lr.slope - tcrit * lr.stderr,
            "slope_ci_high": lr.slope + tcrit * lr.stderr, "intercept": lr.intercept,
            "rmse": float(np.sqrt(np.mean((y - (lr.intercept + lr.slope * x)) ** 2)))}


rel = []
for d in dates + ["mean of both"]:
    x = s2[d]["ndvi"][ok]
    rel.append(relation(x, d0["green"][ok], "Sentinel-2 NDVI", "drone green cover % (random forest)", d))
    rel.append(relation(x, d0["green_thr"][ok], "Sentinel-2 NDVI", "drone green cover % (thresholds)", d))
    rel.append(relation(x, d0["green"][ok] + d0["dry"][ok], "Sentinel-2 NDVI", "drone green + dry cover %", d))
    rel.append(relation(x, d0["bare"][ok], "Sentinel-2 NDVI", "drone bare ground %", d))
    rel.append(relation(x, d0["ndvi"][ok], "Sentinel-2 NDVI", "drone NDVI (from mean red and NIR)", d))
rel.append(relation(d0["ndvi"][ok], d0["green"][ok], "drone NDVI at 10 m", "drone green cover % (random forest)", "drone only"))
rel.append(relation(s2[dates[0]]["ndvi"][ok], s2[dates[1]]["ndvi"][ok], f"Sentinel-2 NDVI {dates[0]}",
                    f"Sentinel-2 NDVI {dates[1]}", "both"))
rel = pd.DataFrame(rel).round(4)
rel.to_csv(TAB / "27_scale_comparison_statistics.csv", index=False)
print(rel[["x", "y", "s2_date", "n", "pearson_r", "p_value", "spearman_r", "slope", "slope_ci_low", "slope_ci_high", "intercept", "rmse"]].to_string())

# site-mean reflectance and NDVI: drone against satellite
lev = [{"source": "Drone (Altum-PT), mean over the same pixels", "red": d0["red"][ok].mean(),
        "nir": d0["nir"][ok].mean(), "ndvi_of_means": float((d0["nir"][ok].mean() - d0["red"][ok].mean()) / (d0["nir"][ok].mean() + d0["red"][ok].mean())),
        "ndvi_sd_between_pixels": d0["ndvi"][ok].std(ddof=1)}]
for d in dates:
    r_, n_ = s2[d]["red"][ok].mean(), s2[d]["nir"][ok].mean()
    lev.append({"source": f"Sentinel-2 L2A {d}", "red": r_, "nir": n_, "ndvi_of_means": (n_ - r_) / (n_ + r_),
                "ndvi_sd_between_pixels": s2[d]["ndvi"][ok].std(ddof=1)})
lev = pd.DataFrame(lev).round(4)
lev.to_csv(TAB / "28_drone_vs_sentinel2_levels.csv", index=False)
print(lev.to_string())

# ---- shift sensitivity -------------------------------------------------------------------
shifts = np.arange(-10, 10.1, 2.5)
sens = []
ndvi_mean = s2["mean of both"]["ndvi"]
with np.errstate(invalid="ignore", divide="ignore"):
    for dy in shifts:
        for dx in shifts:
            dd = aggregate(dx, dy)
            o = dd["ok"]
            if o.sum() < 20:
                continue
            sens.append({"shift_east_m": dx, "shift_north_m": dy, "n": int(o.sum()),
                         "r_green_vs_s2_ndvi": np.corrcoef(ndvi_mean[o], dd["green"][o])[0, 1],
                         "r_drone_ndvi_vs_s2_ndvi": np.corrcoef(ndvi_mean[o], dd["ndvi"][o])[0, 1]})
sens = pd.DataFrame(sens).round(4)
sens.to_csv(TAB / "29_scale_comparison_shift_sensitivity.csv", index=False)
best = sens.loc[sens.r_drone_ndvi_vs_s2_ndvi.idxmax()]
print("best shift for NDVI agreement:", best.to_dict())
print("range of r(green, S2 NDVI) over shifts:", sens.r_green_vs_s2_ndvi.min(), sens.r_green_vs_s2_ndvi.max())

# ---- figures ---------------------------------------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(15, 9.4))
# (a) context map
nd = s2[dates[1]]["ndvi"]
ext = [s2_tr.c, s2_tr.c + s2_shape[1] * 10, s2_tr.f - s2_shape[0] * 10, s2_tr.f]
im = ax[0, 0].imshow(nd, extent=[0, s2_shape[1] * 10, 0, s2_shape[0] * 10], cmap="viridis",
                     vmin=np.percentile(nd, 2), vmax=np.percentile(nd, 98))
from pyproj import Transformer
tf = Transformer.from_crs(crs, s2_crs, always_xy=True)
x0, y0 = tf.transform(tr.c, tr.f + shape[0] * tr.e)
ax[0, 0].add_patch(plt.Rectangle((x0 - ext[0], y0 - ext[2]), shape[1] * tr.a, shape[0] * tr.a, fill=False, ec="red", lw=1.8))
ax[0, 0].set_title(f"(a) Sentinel-2 NDVI, {dates[1]}, 10 m pixels\nred box = drone mosaic (72 m by 70 m)")
ax[0, 0].set_xlabel("Easting in window (m)"); ax[0, 0].set_ylabel("Northing in window (m)")
plt.colorbar(im, ax=ax[0, 0], shrink=0.8, label="NDVI (-)")

# (b) the pixels side by side
rr0, rr1, cc0, cc1 = rows_i.min(), rows_i.max() + 1, cols_i.min(), cols_i.max() + 1
g = np.where(ok, d0["green"], np.nan)[rr0:rr1, cc0:cc1]
im = ax[0, 1].imshow(g, cmap="Greens", extent=[0, (cc1 - cc0) * 10, 0, (rr1 - rr0) * 10])
ax[0, 1].set_title("(b) Drone green cover (%) in Sentinel-2 pixels")
plt.colorbar(im, ax=ax[0, 1], shrink=0.8, label="green cover (%)")
sn = np.where(ok, ndvi_mean, np.nan)[rr0:rr1, cc0:cc1]
im = ax[0, 2].imshow(sn, cmap="viridis", extent=[0, (cc1 - cc0) * 10, 0, (rr1 - rr0) * 10])
ax[0, 2].set_title("(c) Sentinel-2 NDVI, mean of the two dates")
plt.colorbar(im, ax=ax[0, 2], shrink=0.8, label="NDVI (-)")
for a in ax[0, 1:]:
    a.set_xlabel("m"); a.set_ylabel("m")


def scatter(a, x, y, xl, yl, title, one_to_one=False):
    a.scatter(x, y, s=28, c="#0072B2", edgecolor="k", linewidth=0.4)
    lr = stats.linregress(x, y)
    xx = np.linspace(x.min(), x.max(), 50)
    a.plot(xx, lr.intercept + lr.slope * xx, color="#D55E00", lw=1.5)
    if one_to_one:
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        a.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="1:1 line")
        a.legend(fontsize=8, loc="lower right")
    a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=0.3)
    a.set_title(f"{title}\nr = {lr.rvalue:.2f}, p = {lr.pvalue:.3f}, n = {len(x)}")
    a.text(0.03, 0.95, f"y = {lr.intercept:.2f} + {lr.slope:.2f} x", transform=a.transAxes, va="top", fontsize=9)


x = ndvi_mean[ok]
scatter(ax[1, 0], x, d0["green"][ok], "Sentinel-2 NDVI (mean of two dates)", "Drone green cover (%)",
        "(d) Green cover against satellite NDVI")
scatter(ax[1, 1], x, d0["ndvi"][ok], "Sentinel-2 NDVI (mean of two dates)", "Drone NDVI at 10 m",
        "(e) Drone NDVI against satellite NDVI", one_to_one=True)
scatter(ax[1, 2], d0["ndvi"][ok], d0["green"][ok], "Drone NDVI at 10 m", "Drone green cover (%)",
        "(f) Green cover against drone NDVI (same sensor)")
fig.suptitle("Scale comparison: 1.2 cm drone data aggregated to 10 m Sentinel-2 pixels")
fig.tight_layout()
save_fig(fig, FIG / "14_scale_comparison_sentinel2.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for a, col, t in zip(ax, ["r_drone_ndvi_vs_s2_ndvi", "r_green_vs_s2_ndvi"],
                     ["Drone NDVI against Sentinel-2 NDVI", "Drone green cover against Sentinel-2 NDVI"]):
    pv = sens.pivot(index="shift_north_m", columns="shift_east_m", values=col).sort_index(ascending=False)
    im = a.imshow(pv.values, cmap="RdBu_r", vmin=-1, vmax=1,
                  extent=[shifts[0] - 1.25, shifts[-1] + 1.25, shifts[0] - 1.25, shifts[-1] + 1.25])
    a.plot(0, 0, "k+", ms=12)
    a.set_title(t); a.set_xlabel("Shift east (m)"); a.set_ylabel("Shift north (m)")
    plt.colorbar(im, ax=a, shrink=0.85, label="Pearson r")
fig.suptitle("Correlation when the Sentinel-2 grid is shifted against the drone data (+ = no shift)")
fig.tight_layout()
save_fig(fig, FIG / "15_scale_shift_sensitivity.png")
plt.close(fig)
print("step 9 done")
