"""Step 2. Data quality checks, co-registration and artifact mask.

What I do here:
- read the metadata of every raster (CRS, pixel size, size, no-data share);
- put the natural color (RGB) orthomosaic on the multispectral grid by
  averaging, so the two cameras can be compared pixel by pixel;
- measure the shift between the two cameras with phase correlation;
- mask things that are not rangeland surface: the calibration tarps, the
  ground control targets and field gear;
- check for blur and seams with a sharpness map and with row profiles;
- remove the brightness banding between flight lines (a row-wise gain);
- write an overview figure.
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation
from common import (save_fig, RAW, WORK, FIG, TAB, MS_ORTHO, MS_DSM, RGB_ORTHO, RGB_DSM,
                    setup_plot, extent_local, block_mean)

plt = setup_plot()

# ---- 1. metadata table ------------------------------------------------------
rows = []
for name, path in [("Multispectral orthomosaic (5 bands)", MS_ORTHO),
                   ("Multispectral DSM", MS_DSM),
                   ("Natural color orthomosaic (3 bands)", RGB_ORTHO),
                   ("Natural color DSM", RGB_DSM)]:
    with rasterio.open(path) as src:
        b1 = src.read(1)
        nod = src.nodata
        nodata_share = float((b1 == nod).mean()) if nod is not None else float((src.read().sum(0) == 0).mean())
        lon0, lat0, lon1, lat1 = transform_bounds(src.crs, 4326, *src.bounds)
        rows.append({
            "product": name, "file": path.name, "crs": f"EPSG:{src.crs.to_epsg()}",
            "pixel_size_cm": round(src.res[0] * 100, 3),
            "width_px": src.width, "height_px": src.height,
            "width_m": round(src.width * src.res[0], 2),
            "height_m": round(src.height * src.res[1], 2),
            "bands": src.count, "dtype": src.dtypes[0],
            "nodata_share_pct": round(100 * nodata_share, 4),
            "west": round(lon0, 6), "south": round(lat0, 6),
            "east": round(lon1, 6), "north": round(lat1, 6)})
meta = pd.DataFrame(rows)
meta.to_csv(TAB / "01_raster_metadata.csv", index=False)
print(meta.to_string())

# ---- 2. read multispectral, put RGB on the same grid -------------------------
with rasterio.open(MS_ORTHO) as src:
    ms = src.read()                       # blue, green, red, red edge, NIR (reflectance)
    tr, crs = src.transform, src.crs
    nodata = src.nodata
shape = ms.shape[1:]
valid = np.all(ms != nodata, axis=0) & np.all(np.isfinite(ms), axis=0)

rgb = np.zeros((3,) + shape, "float32")
with rasterio.open(RGB_ORTHO) as src:
    for i in range(3):
        reproject(rasterio.band(src, i + 1), rgb[i], dst_transform=tr, dst_crs=crs,
                  resampling=Resampling.average)

# ---- 2b. brightness banding between flight lines -------------------------------
# The flight lines run east to west, so a change in sunlight between lines
# shows up as a change of brightness from row to row. I measure the median of
# every image row (72 m wide, mostly soil) and divide by its smoothed value
# relative to the site median. For the multispectral camera one common gain is
# used for all five bands, so band ratios such as NDVI do not change.
# For the RGB camera each band gets its own gain because its blue channel drifts.
def row_gain(band):
    med = np.median(band[:, ::5], axis=1)
    med = ndi.uniform_filter1d(med, size=83, mode="nearest")     # about 1 m
    return med / np.median(med)


ms_gain_bands = np.stack([row_gain(ms[i]) for i in range(5)])
ms_gain = np.median(ms_gain_bands, axis=0)
rgb_gain = np.stack([row_gain(rgb[i]) for i in range(3)])
ms_raw_rowmed = ms_gain_bands.copy()
ms = (ms / ms_gain[None, :, None]).astype("float32")
rgb = (rgb / rgb_gain[:, :, None]).astype("float32")
np.save(WORK / "ms_corrected.npy", ms)
np.save(WORK / "rgb_on_ms.npy", rgb)
ms_after = np.stack([row_gain(ms[i]) for i in range(5)])
rgb_after = np.stack([row_gain(rgb[i]) for i in range(3)])

# band statistics
q = [1, 5, 25, 50, 75, 95, 99]
names = ["blue_475nm", "green_560nm", "red_668nm", "rededge_717nm", "nir_842nm"]
st = pd.DataFrame([np.percentile(ms[i][valid][::13], q) for i in range(5)],
                  index=names, columns=[f"p{k}" for k in q]).round(4)
st.to_csv(TAB / "02_band_reflectance_percentiles.csv")
print(st)

# ---- 3. co-registration check between the two cameras -------------------------
reg = []
for r in (400, 2600, 4800):
    for c in (400, 2700, 5000):
        a = ms[1, r:r + 600, c:c + 600]
        b = rgb[1, r:r + 600, c:c + 600]
        shift, _, _ = phase_cross_correlation(a, b, upsample_factor=10)
        reg.append({"window_row": r, "window_col": c,
                    "shift_rows_px": shift[0], "shift_cols_px": shift[1],
                    "shift_cm": float(np.hypot(*shift) * tr.a * 100),
                    "pearson_r_green": float(np.corrcoef(a.ravel(), b.ravel())[0, 1])})
reg = pd.DataFrame(reg).round(3)
reg.to_csv(TAB / "03_coregistration_check.csv", index=False)
print(reg)

# ---- 4. artifact mask ----------------------------------------------------------
# The soil here is red and the plants are dull, so large patches that are
# neutral grey, white or black are man-made: tarps, targets, gear.
R, G, B = rgb
V = (R + G + B) / 3.0
sat = (np.maximum(np.maximum(R, G), B) - np.minimum(np.minimum(R, G), B)) / (np.maximum(np.maximum(R, G), B) + 1e-6)
# Smooth over about 10 cm so single grey leaves are not picked up.
k = 9
sat_s = ndi.uniform_filter(sat, k)
V_s = ndi.uniform_filter(V, k)
cand = (sat_s < 0.15) | (V_s > 190)
lab, n = ndi.label(cand)
sizes = ndi.sum(cand, lab, np.arange(1, n + 1))
keep_ids = np.where(sizes * tr.a ** 2 > 0.01)[0] + 1      # larger than 0.01 m2
art = np.isin(lab, keep_ids)
art = ndi.binary_dilation(art, iterations=40)              # about 50 cm safety buffer
# The six ground control targets have surveyed positions, so I mask them directly too.
gcp = pd.read_csv(RAW / "2023-05_Moab_UT_UAS_GCPs.csv")
gcp = gcp[gcp["ID"].str.startswith("Site7")]
rr, cc = np.mgrid[0:shape[0], 0:shape[1]]
for _, g in gcp.iterrows():
    c0, r0 = ~tr * (g["Easting_UTM12N"], g["Northing_UTM12N"])
    art |= (np.abs(rr - r0) < 0.6 / tr.a) & (np.abs(cc - c0) < 0.6 / tr.a)
del rr, cc
mask = valid & ~art
np.save(WORK / "valid_mask.npy", mask)
print("artifact share %:", round(100 * art.mean(), 3), " components:", len(keep_ids))

# ---- 5. sharpness (blur and seam check) ---------------------------------------
lap = ndi.laplace(ndi.gaussian_filter(ms[1], 1.0))
lap[~mask] = np.nan
cell = int(round(2.0 / tr.a))                 # 2 m cells
sharp = np.sqrt(block_mean(lap ** 2, cell))
sharp_rel = sharp / np.nanmedian(sharp)

qa = pd.DataFrame([{
    "valid_pixels": int(valid.sum()), "nodata_share_pct": round(100 * (1 - valid.mean()), 4),
    "artifact_mask_share_pct": round(100 * art.mean(), 3),
    "analysis_area_m2": round(mask.sum() * tr.a ** 2, 1),
    "median_shift_between_cameras_cm": round(reg["shift_cm"].median(), 2),
    "max_shift_between_cameras_cm": round(reg["shift_cm"].max(), 2),
    "sharpness_2m_cells_min_rel_to_median": round(float(np.nanmin(sharp_rel)), 3),
    "sharpness_2m_cells_p05_rel_to_median": round(float(np.nanpercentile(sharp_rel, 5)), 3),
    "sharpness_2m_cells_cv": round(float(np.nanstd(sharp) / np.nanmean(sharp)), 3),
    "ms_row_gain_min": round(float(ms_gain.min()), 3),
    "ms_row_gain_max": round(float(ms_gain.max()), 3),
    "ms_row_gain_max_spread_between_bands": round(float(np.ptp(ms_raw_rowmed, axis=0).max()), 3),
    "rgb_row_gain_min": round(float(rgb_gain.min()), 3),
    "rgb_row_gain_max": round(float(rgb_gain.max()), 3),
    "ms_row_gain_after_min": round(float(ms_after.min()), 3),
    "ms_row_gain_after_max": round(float(ms_after.max()), 3),
}]).T
qa.columns = ["value"]
qa.to_csv(TAB / "04_qa_summary.csv")
print(qa)

# ---- 6. overview figure ---------------------------------------------------------
ext = extent_local(tr, shape)
s = 4
fig, ax = plt.subplots(2, 2, figsize=(11, 10.5))
ax[0, 0].imshow(np.clip(np.moveaxis(rgb[:, ::s, ::s], 0, -1) / 190.0, 0, 1), extent=ext)
ax[0, 0].set_title("(a) Natural color orthomosaic (Skydio X2D, 0.77 cm)")
cir = np.clip(np.stack([ms[4], ms[2], ms[1]], -1)[::s, ::s] / 0.45, 0, 1)
ax[0, 1].imshow(cir, extent=ext)
ax[0, 1].set_title("(b) False color: NIR, red, green (Altum-PT, 1.2 cm)")
show = np.clip(np.moveaxis(rgb[:, ::s, ::s], 0, -1) / 190.0, 0, 1).copy()
show[art[::s, ::s]] = [0.0, 0.45, 0.70]
ax[1, 0].imshow(show, extent=ext)
ax[1, 0].set_title(f"(c) Masked artifacts in blue ({100 * art.mean():.2f}% of area)")
im = ax[1, 1].imshow(sharp_rel, extent=ext, cmap="cividis", vmin=0.6, vmax=1.4)
ax[1, 1].set_title("(d) Sharpness of the green band, 2 m cells\n(relative to the site median)")
plt.colorbar(im, ax=ax[1, 1], shrink=0.8, label="relative sharpness (-)")
for a in ax.ravel():
    a.set_xlabel("Easting from SW corner (m)")
    a.set_ylabel("Northing from SW corner (m)")
fig.suptitle("Moab site 7, drone survey of 5 May 2023 (NAD83(2011) / UTM zone 12N, EPSG:6341)", y=0.995)
fig.tight_layout()
save_fig(fig, FIG / "01_overview_qa.png")
plt.close(fig)

# ---- 7. row profile figure (banding before and after) ----------------------------
north = (shape[0] - np.arange(shape[0])) * tr.a
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
cols5 = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#000000"]
for i, n in enumerate(["blue", "green", "red", "red edge", "NIR"]):
    ax[0].plot(north, ms_raw_rowmed[i], color=cols5[i], lw=0.9, label=n)
    ax[2].plot(north, ms_after[i], color=cols5[i], lw=0.9, label=n)
for i, n in enumerate(["red", "green", "blue"]):
    ax[1].plot(north, rgb_gain[i], color=["#D55E00", "#009E73", "#0072B2"][i], lw=0.9, label=n)
ax[0].set_title("(a) Multispectral camera, as delivered")
ax[1].set_title("(b) Natural color camera, as delivered")
ax[2].set_title("(c) Multispectral camera, after row gain")
for a in ax:
    a.set_xlabel("Northing from SW corner (m)")
    a.grid(alpha=0.3)
    a.legend(fontsize=8, ncol=2)
ax[0].set_ylabel("Row median / site median (-)")
fig.tight_layout()
save_fig(fig, FIG / "02_brightness_banding.png")
plt.close(fig)
print("step 2 done")
