"""Step 5. Build the labeled reference points for training and testing.

The data release has no labeled pixels, so I build the labels by a fixed,
documented rule. To keep the labels as independent as possible from the
classifier inputs, the labels come from the natural color camera (Skydio X2D,
a different aircraft, flight and sensor), while the classifiers only see the
multispectral camera (Altum-PT) and its DSM.

Rule (applied to the natural color mosaic averaged to the 1.2 cm grid):
  V = (R + G + B) / 3 in digital numbers, exg = (2G - R - B) / (R + G + B),
  rb = (R - B) / (R + B)
  green vegetation         exg > 0.04 and V >= 50
  dry vegetation / litter  rb < 0.31 and 55 <= V < 110 and exg < 0.02  (grey, dull)
  bare ground              rb > 0.37 and V > 95 and exg < 0.005         (bright, red)
  shadow / other           V < 45 and also dark in the multispectral mosaic
                           (mean reflectance < 0.09)
Everything else is left unlabeled (mixed or unclear pixels).
A point is kept only if all pixels in its 3 x 3 neighborhood (3.6 cm) have
the same label. This protects against the small shift between the two
cameras (about 0.5 cm, see step 2).

The thresholds were set once by looking at the color histograms and at image
chips (figure 07). They were not tuned to make any classifier look better.

Shadow is the one class that cannot be labeled from the other camera alone,
because the two flights were at different times and the shadows moved.
For that class the label also uses the multispectral brightness, so its
accuracy numbers are not independent. I say so in the README.

Spatial split: the site is cut into 10 m blocks. Blocks are assigned at
random to training (60%) or testing (40%). Points are drawn at random within
each class and block group, so no test point lies in a training block.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from matplotlib.colors import ListedColormap
from common import (save_fig, WORK, DATA, FIG, TAB, SEED, CLASSES, CLASS_COLORS, setup_plot, ms_grid)

plt = setup_plot()
tr, crs, shape = ms_grid()
rgb = np.load(WORK / "rgb_on_ms.npy")
mask = np.load(WORK / "valid_mask.npy")
ms_bright = np.load(WORK / "ms_corrected.npy").mean(axis=0)

R, G, B = rgb
V = (R + G + B) / 3.0
exg = (2 * G - R - B) / (R + G + B + 1e-6)
rb = (R - B) / (R + B + 1e-6)

lab = np.zeros(shape, "uint8")
lab[(rb > 0.37) & (V > 95) & (exg < 0.005)] = 3
lab[(rb < 0.31) & (V >= 55) & (V < 110) & (exg < 0.02)] = 2
lab[(exg > 0.04) & (V >= 50)] = 1
lab[(V < 45) & (ms_bright < 0.09)] = 4
lab[~mask] = 0

pure = np.zeros(shape, "uint8")
for k in CLASSES:
    pure[ndi.minimum_filter((lab == k).astype("uint8"), size=3) == 1] = k

# ---- blocks and split ----------------------------------------------------------
block_px = int(round(10.0 / tr.a))
rows, cols = np.indices(shape, sparse=True)
block_id = (rows // block_px) * 100 + (cols // block_px)
ids = np.unique(block_id)
rng = np.random.default_rng(SEED)
perm = rng.permutation(ids)
n_test = int(round(0.4 * len(ids)))
test_blocks = set(perm[:n_test].tolist())
is_test = np.isin(block_id, list(test_blocks))
is_test = np.broadcast_to(is_test, shape)

# ---- stratified random sample ----------------------------------------------------
N_TRAIN, N_TEST = 3000, 2000
pts = []
for k in CLASSES:
    for split, sel, n in (("train", ~is_test, N_TRAIN), ("test", is_test, N_TEST)):
        rr, cc = np.nonzero((pure == k) & sel)
        take = rng.choice(len(rr), size=min(n, len(rr)), replace=False)
        for r_, c_ in zip(rr[take], cc[take]):
            x, y = tr * (c_ + 0.5, r_ + 0.5)
            pts.append((r_, c_, round(x, 3), round(y, 3), k, CLASSES[k],
                        int((r_ // block_px) * 100 + (c_ // block_px)), split))
pts = pd.DataFrame(pts, columns=["row", "col", "easting_m", "northing_m", "class_id",
                                 "class_name", "block_id", "split"])
pts.to_csv(DATA / "reference_points.csv", index=False)
np.save(WORK / "reference_pure.npy", pure)
np.save(WORK / "block_is_test.npy", np.packbits(is_test))

n_valid = mask.sum()
summ = []
for k, name in CLASSES.items():
    summ.append({"class_id": k, "class_name": name,
                 "rule_labeled_share_pct": 100 * (lab == k).sum() / n_valid,
                 "pure_3x3_share_pct": 100 * (pure == k).sum() / n_valid,
                 "pure_pixels": int((pure == k).sum()),
                 "train_points": int(((pts.class_id == k) & (pts.split == "train")).sum()),
                 "test_points": int(((pts.class_id == k) & (pts.split == "test")).sum())})
summ.append({"class_id": 0, "class_name": "unlabeled (mixed or unclear)",
             "rule_labeled_share_pct": 100 * ((lab == 0) & mask).sum() / n_valid,
             "pure_3x3_share_pct": 100 * ((pure == 0) & mask).sum() / n_valid,
             "pure_pixels": int(((pure == 0) & mask).sum()), "train_points": 0, "test_points": 0})
summ = pd.DataFrame(summ).round(3)
summ.to_csv(TAB / "11_reference_label_summary.csv", index=False)
print(summ.to_string())
print("blocks:", len(ids), "test blocks:", n_test,
      "train blocks with points:", pts[pts.split == "train"].block_id.nunique(),
      "test blocks with points:", pts[pts.split == "test"].block_id.nunique())

# ---- figure: chips with labels, and the block split -----------------------------------
cmap = ListedColormap(["#FFFFFF"] + [CLASS_COLORS[k] for k in CLASSES])
fig = plt.figure(figsize=(13, 8.6))
chips = [(700, 900), (2700, 2800), (4800, 1500)]
n = 330                                             # 4 m
for i, (r0, c0) in enumerate(chips):
    a = fig.add_subplot(2, 4, i + 1)
    a.imshow(np.clip(np.moveaxis(rgb[:, r0:r0 + n, c0:c0 + n], 0, -1) / 190, 0, 1),
             extent=[0, n * tr.a, 0, n * tr.a], interpolation="nearest")
    a.set_title(f"Natural color, chip {i + 1}")
    a.set_xlabel("m"); a.set_ylabel("m")
    b = fig.add_subplot(2, 4, i + 5)
    b.imshow(lab[r0:r0 + n, c0:c0 + n], cmap=cmap, vmin=0, vmax=4,
             extent=[0, n * tr.a, 0, n * tr.a], interpolation="nearest")
    b.set_title(f"Rule labels, chip {i + 1}")
    b.set_xlabel("m"); b.set_ylabel("m")
a = fig.add_subplot(2, 4, 4)
blk = np.where(is_test[::20, ::20], 1.0, 0.0)
a.imshow(blk, cmap=ListedColormap(["#56B4E9", "#F0E442"]), extent=[0, shape[1] * tr.a, 0, shape[0] * tr.a])
sub = pts.sample(3000, random_state=1)
a.scatter(sub.col * tr.a, (shape[0] - sub.row) * tr.a, s=0.6, c="k")
a.set_title("10 m blocks: blue = train, yellow = test\n(dots: 3,000 of the reference points)")
a.set_xlabel("Easting from SW corner (m)"); a.set_ylabel("Northing from SW corner (m)")
a = fig.add_subplot(2, 4, 8)
a.axis("off")
handles = [plt.Rectangle((0, 0), 1, 1, fc=CLASS_COLORS[k], ec="k") for k in CLASSES]
handles.append(plt.Rectangle((0, 0), 1, 1, fc="#FFFFFF", ec="k"))
a.legend(handles, list(CLASSES.values()) + ["Unlabeled (mixed or unclear)"], loc="center", frameon=False,
         title="Reference label")
fig.suptitle("Reference labels from the natural color camera (4 m by 4 m chips) and the spatial split")
fig.tight_layout()
save_fig(fig, FIG / "07_reference_labels.png")
plt.close(fig)
print("step 5 done")
