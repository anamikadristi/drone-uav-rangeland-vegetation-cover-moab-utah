"""Step 6. Classify every pixel with two methods.

Method A, index thresholds (no training data):
  1. shadow / other         if NIR reflectance < 0.10 (a fixed physical cut)
  2. green vegetation       if NDVI > Otsu threshold of NDVI
  3. bare ground            if soil redness > Otsu threshold of redness
  4. dry vegetation/litter  everything else
  Otsu's method picks the cut that best splits a histogram into two groups.

Method B, random forest (supervised):
  A random forest is a set of decision trees, each grown on a random
  resample of the training points; the class is decided by majority vote.
  Inputs (all from the multispectral flight): 5 bands, NDVI, NDRE, MSAVI,
  ExG, soil redness, mean brightness, local standard deviation of NIR and of
  NDVI (5 x 5 pixels), local mean NDVI (9 x 9 pixels) and height above ground
  from the multispectral DSM.

Both maps are written as GeoTIFF (EPSG:6341). Accuracy is computed in step 7.
"""
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
from common import WORK, DATA, TAB, MAPS, SEED, CLASSES, ms_grid

tr, crs, shape = ms_grid()
mask = np.load(WORK / "valid_mask.npy")
ms = np.load(WORK / "ms_corrected.npy")
idx = np.load(WORK / "indices.npz")
height = np.load(WORK / "height_ms_dsm.npy")
ndvi, red_i = idx["NDVI"], idx["REDNESS"]


def write_map(arr, path):
    prof = dict(driver="GTiff", height=shape[0], width=shape[1], count=1, dtype="uint8",
                crs=crs, transform=tr, compress="deflate", predictor=2, tiled=True,
                blockxsize=512, blockysize=512, nodata=0)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)
        dst.update_tags(classes="1 green vegetation; 2 dry vegetation or litter; 3 bare ground; "
                                "4 shadow or other; 0 masked")


# ---- Method A: thresholds -----------------------------------------------------------
sub = (slice(None, None, 4), slice(None, None, 4))
t_ndvi = float(threshold_otsu(ndvi[sub][mask[sub]]))
t_red = float(threshold_otsu(red_i[sub][mask[sub]]))
T_NIR = 0.10
thr = np.full(shape, 2, "uint8")
thr[red_i > t_red] = 3
thr[ndvi > t_ndvi] = 1
thr[ms[4] < T_NIR] = 4
thr[~mask] = 0
write_map(thr, MAPS / "class_threshold.tif")
pd.DataFrame([{"otsu_ndvi": t_ndvi, "otsu_redness": t_red, "nir_shadow_cut": T_NIR}]).round(4).to_csv(
    TAB / "12_threshold_values.csv", index=False)
print("thresholds: NDVI", round(t_ndvi, 4), "redness", round(t_red, 4))

# ---- Method B: features ---------------------------------------------------------------
def local_std(a, k):
    m = ndi.uniform_filter(a, k)
    m2 = ndi.uniform_filter(a * a, k)
    return np.sqrt(np.maximum(m2 - m * m, 0))


FEATURES = ["blue", "green", "red", "rededge", "nir", "NDVI", "NDRE", "MSAVI", "ExG", "REDNESS",
            "brightness", "nir_std5", "ndvi_std5", "ndvi_mean9", "height_m"]
feat = [ms[0], ms[1], ms[2], ms[3], ms[4], ndvi, idx["NDRE"], idx["MSAVI"], idx["ExG"], red_i,
        ms.mean(axis=0), local_std(ms[4], 5), local_std(ndvi, 5), ndi.uniform_filter(ndvi, 9), height]
feat = [f.astype("float32") for f in feat]

pts = pd.read_csv(DATA / "reference_points.csv")
X = np.stack([f[pts.row.values, pts.col.values] for f in feat], axis=1)
y = pts.class_id.values
tr_i = (pts.split == "train").values
te_i = ~tr_i

rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, max_features="sqrt",
                            class_weight="balanced", n_jobs=-1, random_state=SEED, oob_score=True)
rf.fit(X[tr_i], y[tr_i])
print("RF out-of-bag accuracy (train blocks):", round(rf.oob_score_, 4))
print("RF accuracy on test blocks:", round((rf.predict(X[te_i]) == y[te_i]).mean(), 4))

# feature importance: impurity based, and permutation on the test blocks
perm = permutation_importance(rf, X[te_i], y[te_i], n_repeats=10, random_state=SEED, n_jobs=-1)
imp = pd.DataFrame({"feature": FEATURES, "gini_importance": rf.feature_importances_,
                    "permutation_importance_test": perm.importances_mean,
                    "permutation_sd": perm.importances_std}).sort_values(
    "permutation_importance_test", ascending=False).round(4)
imp.to_csv(TAB / "13_rf_feature_importance.csv", index=False)
print(imp.to_string())

# spatial 5-fold cross-validation over the 10 m blocks, as a second check
cv = []
for f, (a, b) in enumerate(GroupKFold(n_splits=5).split(X, y, groups=pts.block_id.values)):
    m = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, max_features="sqrt",
                               class_weight="balanced", n_jobs=-1, random_state=SEED + f)
    m.fit(X[a], y[a])
    p = m.predict(X[b])
    cv.append({"fold": f + 1, "n_blocks": len(set(pts.block_id.values[b])),
               "n_points": len(b), "overall_accuracy": (p == y[b]).mean(),
               "balanced_accuracy": np.mean([(p[y[b] == k] == k).mean() for k in CLASSES])})
cv = pd.DataFrame(cv).round(4)
cv.to_csv(TAB / "14_rf_spatial_cv.csv", index=False)
print(cv.to_string())

# a reduced model with only the three visible bands and visible indices, to show
# what an RGB-only drone could do at this site
vis = [FEATURES.index(k) for k in ("blue", "green", "red", "ExG", "REDNESS")]
rf_vis = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, max_features="sqrt",
                                class_weight="balanced", n_jobs=-1, random_state=SEED)
rf_vis.fit(X[tr_i][:, vis], y[tr_i])
pd.DataFrame({"row": pts.row[te_i], "col": pts.col[te_i], "class_id": y[te_i],
              "rf_visible_only": rf_vis.predict(X[te_i][:, vis])}).to_csv(
    WORK / "rf_visible_only_test_predictions.csv", index=False)

# ---- predict the whole mosaic in chunks ---------------------------------------------------
out = np.zeros(shape, "uint8")
step = 300
for r0 in range(0, shape[0], step):
    sl = slice(r0, min(r0 + step, shape[0]))
    Xc = np.stack([f[sl].ravel() for f in feat], axis=1)
    out[sl] = rf.predict(Xc).reshape(-1, shape[1])
out[~mask] = 0
write_map(out, MAPS / "class_random_forest.tif")

# ---- how much does the site cover move if the training data change? ----------------------
# I refit a smaller forest 30 times, each time on training blocks resampled with
# replacement, and predict a regular 1-in-49 sample of pixels (every 7th row and column).
ss = (slice(3, None, 7), slice(3, None, 7))
ms_s = mask[ss].ravel()
Xs = np.stack([f[ss].ravel() for f in feat], axis=1)[ms_s]
tr_blocks = pts.block_id.values[tr_i]
ub = np.unique(tr_blocks)
rng = np.random.default_rng(SEED)
Xtr, ytr = X[tr_i], y[tr_i]
rep = []
for b in range(30):
    pick = rng.choice(ub, len(ub))
    ii = np.concatenate([np.nonzero(tr_blocks == k)[0] for k in pick])
    m = RandomForestClassifier(n_estimators=60, min_samples_leaf=3, max_features="sqrt",
                               class_weight="balanced", n_jobs=-1, random_state=SEED + 100 + b)
    m.fit(Xtr[ii], ytr[ii])
    p = m.predict(Xs)
    rep.append({"replicate": b + 1, **{f"cover_pct_{k}": 100 * (p == k).mean() for k in CLASSES}})
rep = pd.DataFrame(rep).round(3)
rep.to_csv(TAB / "19_site_cover_model_bootstrap.csv", index=False)
print(rep.describe().loc[["mean", "std", "min", "max"]].round(2).to_string())

for name, arr in (("threshold", thr), ("random forest", out)):
    share = {CLASSES[k]: round(100 * (arr == k).sum() / mask.sum(), 2) for k in CLASSES}
    print(name, share)
print("step 6 done")
