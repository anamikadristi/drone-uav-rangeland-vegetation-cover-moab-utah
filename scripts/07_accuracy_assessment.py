"""Step 7. Accuracy assessment on the spatially separate test blocks.

For each method I build the confusion matrix (rows = map class,
columns = reference class) from the test points and report:
  overall accuracy     share of test points with the right class
  producer's accuracy  of the reference points of a class, the share mapped
                       correctly (1 - omission error); same as recall
  user's accuracy      of the points mapped as a class, the share that really
                       are that class (1 - commission error); same as precision
  F1                   harmonic mean of user's and producer's accuracy
  kappa                agreement beyond what chance would give
The test sample has the same number of points for the three main classes,
which is not how common they are on the ground. So I also report an overall
accuracy weighted by each class's share among the clear reference pixels.

Confidence intervals: block bootstrap. I resample the 22 test blocks with
replacement 2,000 times and recompute every metric, so the intervals reflect
that points inside a block are not independent.
"""
import numpy as np
import pandas as pd
import rasterio
from common import save_fig, WORK, DATA, TAB, FIG, MAPS, SEED, CLASSES, setup_plot

plt = setup_plot()
K = list(CLASSES)
pts = pd.read_csv(DATA / "reference_points.csv")
te = pts[pts.split == "test"].reset_index(drop=True)
maps = {}
for name, f in (("Index thresholds", "class_threshold.tif"), ("Random forest", "class_random_forest.tif")):
    with rasterio.open(MAPS / f) as src:
        maps[name] = src.read(1)[te.row.values, te.col.values]
vis = pd.read_csv(WORK / "rf_visible_only_test_predictions.csv")
assert (vis.row.values == te.row.values).all()
maps["Random forest, visible bands only"] = vis.rf_visible_only.values

# class weights: share of each class among the clear (3 x 3 pure) reference pixels in test blocks
pure = np.load(WORK / "reference_pure.npy")
shape = pure.shape
is_test = np.unpackbits(np.load(WORK / "block_is_test.npy"))[:pure.size].reshape(shape).astype(bool)
w = np.array([((pure == k) & is_test).sum() for k in K], float)
w = w / w.sum()
del pure, is_test
ref = te.class_id.values
blocks = te.block_id.values
ublocks = np.unique(blocks)
by_block = {b: np.nonzero(blocks == b)[0] for b in ublocks}


def confusion(pred, refv):
    cm = np.zeros((len(K), len(K)), int)
    for i, a in enumerate(K):
        for j, b in enumerate(K):
            cm[i, j] = ((pred == a) & (refv == b)).sum()
    return cm


def metrics(cm):
    n = cm.sum()
    oa = np.trace(cm) / n
    with np.errstate(invalid="ignore", divide="ignore"):
        ua = np.diag(cm) / cm.sum(axis=1)
        pa = np.diag(cm) / cm.sum(axis=0)
        f1 = 2 * ua * pa / (ua + pa)
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / n ** 2
    kappa = (oa - pe) / (1 - pe)
    oa_w = np.nansum(w * pa)            # overall accuracy weighted by reference class share
    out = {"overall_accuracy": oa, "kappa": kappa, "area_weighted_overall_accuracy": oa_w,
           "macro_f1": np.nanmean(f1)}
    for i, k in enumerate(K):
        out[f"producers_{k}"] = pa[i]
        out[f"users_{k}"] = ua[i]
        out[f"f1_{k}"] = f1[i]
    return out


rng = np.random.default_rng(SEED)
NB = 2000
boot_sets = [np.concatenate([by_block[b] for b in rng.choice(ublocks, len(ublocks))]) for _ in range(NB)]
summary, perclass, cms = [], [], {}
boot_store = {}
for name, pred in maps.items():
    cm = confusion(pred, ref)
    cms[name] = cm
    m = metrics(cm)
    boots = pd.DataFrame([metrics(confusion(pred[i], ref[i])) for i in boot_sets])
    boot_store[name] = boots
    lo, hi = boots.quantile(0.025), boots.quantile(0.975)
    row = {"method": name, "n_test_points": len(ref), "n_test_blocks": len(ublocks)}
    for k in ("overall_accuracy", "area_weighted_overall_accuracy", "kappa", "macro_f1"):
        row[k] = m[k]; row[k + "_ci_low"] = lo[k]; row[k + "_ci_high"] = hi[k]
    summary.append(row)
    for kk in K:
        r = {"method": name, "class": CLASSES[kk], "n_reference": int(cm[:, K.index(kk)].sum()),
             "n_mapped": int(cm[K.index(kk)].sum())}
        for nm, key in (("producers_accuracy", f"producers_{kk}"), ("users_accuracy", f"users_{kk}"), ("f1", f"f1_{kk}")):
            r[nm] = m[key]; r[nm + "_ci_low"] = lo[key]; r[nm + "_ci_high"] = hi[key]
        perclass.append(r)
    pd.DataFrame(cm, index=[f"map: {CLASSES[k]}" for k in K],
                 columns=[f"ref: {CLASSES[k]}" for k in K]).to_csv(
        TAB / f"17_confusion_{name.lower().replace(' ', '_').replace(',', '')}.csv")
summary = pd.DataFrame(summary).round(4)
perclass = pd.DataFrame(perclass).round(4)
summary.to_csv(TAB / "15_accuracy_summary.csv", index=False)
perclass.to_csv(TAB / "16_accuracy_per_class.csv", index=False)
print(summary.T.to_string())
print(perclass.to_string())

# paired difference random forest minus thresholds, with the same bootstrap samples
d = boot_store["Random forest"] - boot_store["Index thresholds"]
diff = pd.DataFrame({"metric": ["overall_accuracy", "area_weighted_overall_accuracy", "kappa", "macro_f1"]})
diff["rf_minus_threshold"] = [summary.loc[1, k] - summary.loc[0, k] for k in diff.metric]
diff["ci_low"] = [d[k].quantile(0.025) for k in diff.metric]
diff["ci_high"] = [d[k].quantile(0.975) for k in diff.metric]
diff.round(4).to_csv(TAB / "18_accuracy_difference_rf_minus_threshold.csv", index=False)
print(diff.round(4).to_string())
print("class weights used:", dict(zip([CLASSES[k] for k in K], w.round(4))))

# ---- figure: confusion matrices -----------------------------------------------------------
short = ["Green", "Dry/litter", "Bare", "Shadow"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
for a, (name, cm) in zip(ax, cms.items()):
    frac = cm / cm.sum(axis=0, keepdims=True)
    a.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            a.text(j, i, f"{cm[i, j]}\n{100 * frac[i, j]:.0f}%", ha="center", va="center",
                   color="white" if frac[i, j] > 0.55 else "black", fontsize=9)
    a.set_xticks(range(4)); a.set_xticklabels(short)
    a.set_yticks(range(4)); a.set_yticklabels(short)
    a.set_xlabel("Reference class (natural color camera rule)")
    a.set_ylabel("Mapped class")
    s = summary[summary.method == name].iloc[0]
    a.set_title(f"{name}\nOA {100 * s.overall_accuracy:.1f}% "
                f"({100 * s.overall_accuracy_ci_low:.1f} to {100 * s.overall_accuracy_ci_high:.1f}), "
                f"kappa {s.kappa:.2f}")
fig.suptitle("Confusion matrices on 22 test blocks (counts and share of each reference class)")
fig.tight_layout()
save_fig(fig, FIG / "08_confusion_matrices.png")
plt.close(fig)

# ---- figure: per-class accuracy with intervals ------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
cols = {"Index thresholds": "#E69F00", "Random forest": "#0072B2", "Random forest, visible bands only": "#999999"}
for a, key, lab in zip(ax, ["producers_accuracy", "users_accuracy", "f1"],
                       ["Producer's accuracy", "User's accuracy", "F1 score"]):
    for i, name in enumerate(maps):
        d_ = perclass[perclass.method == name]
        x = np.arange(4) + (i - 1) * 0.26
        v = d_[key].values
        err = np.vstack([v - d_[key + "_ci_low"].values, d_[key + "_ci_high"].values - v])
        a.bar(x, v, 0.25, yerr=np.nan_to_num(err), color=cols[name], label=name, capsize=2)
    a.set_xticks(range(4)); a.set_xticklabels(short)
    a.set_title(lab); a.set_ylim(0, 1.05); a.grid(axis="y", alpha=0.3)
ax[0].set_ylabel("value (0 to 1)")
ax[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Per-class accuracy on test blocks; bars show 95% block-bootstrap intervals")
fig.tight_layout()
save_fig(fig, FIG / "09_per_class_accuracy.png")
plt.close(fig)
print("step 7 done")
