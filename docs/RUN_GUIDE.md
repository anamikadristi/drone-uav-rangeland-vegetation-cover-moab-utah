# Run guide

Author: Anamika Dristi, Ph.D.

Step-by-step instructions to rerun everything on a new computer.

## 1. What you need

- Python 3.12 (3.11 should also work).
- About 4 GB of free disk space (1.2 GB of downloads and about 2.7 GB of intermediate arrays).
- About 8 GB of memory. The multispectral mosaic is held in memory as a 690 MB array, and a few steps hold several arrays of that size.
- An internet connection for step 1. No account or key is needed.
- No GIS program and no photogrammetry program is needed.

## 2. Install

```bash
git clone <this repository>
cd drone-uav-rangeland-vegetation-cover-moab-utah
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` lists the versions used for the results in the README. With conda, `conda install -c conda-forge rasterio laspy lazrs-python scikit-image scikit-learn` works as well.

## 3. Run everything

```bash
bash scripts/run_all.sh
```

Optional settings:

```bash
PYTHON=/path/to/python bash scripts/run_all.sh          # use a specific Python
DRONE_DATA_DIR=/big/disk/moab bash scripts/run_all.sh   # keep the large files elsewhere
```

By default the large files go to `data/raw/` and the intermediate arrays to `data/raw/work/`. Both are ignored by git.

## 4. Run step by step

Run the scripts from the `scripts/` folder, in order.

| Step | Script | What it does | Time on a laptop |
|---|---|---|---|
| 1 | `01_download_data.py` | Downloads the USGS drone files (1.2 GB). Skips files already present. Skips the BLM record and Sentinel-2 windows if they are already in `data/`. | 2 to 20 minutes, depends on the connection |
| 2 | `02_data_qa.py` | Metadata, co-registration, artifact mask, row gain, overview figures | about 40 s |
| 3 | `03_vegetation_indices.py` | Nine indices, histograms, maps | about 25 s |
| 4 | `04_vegetation_height.py` | Height from two DSMs and lidar, slope | about 30 s |
| 5 | `05_reference_labels.py` | Rule labels, 10 m blocks, reference points | about 60 s |
| 6 | `06_classify.py` | Threshold map, random forest map, importance, cross-validation, 30 refits | about 90 s |
| 7 | `07_accuracy_assessment.py` | Confusion matrices, metrics, bootstrap intervals | about 30 s |
| 8 | `08_fractional_cover.py` | Fractional cover, site cover, field check, bare patches | about 30 s |
| 9 | `09_scale_comparison.py` | Sentinel-2 comparison and shift test | about 50 s |

Steps 2 to 9 together took 5 minutes 18 seconds on my laptop (Apple silicon, 8 cores).

Each script reads what the earlier ones wrote, so a script can be rerun alone once the earlier steps have been run.

## 5. How to check the result

After the run, compare with these values from `outputs/tables/`:

| File | Value to check | Expected |
|---|---|---|
| `01_raster_metadata.csv` | pixel size of the multispectral mosaic | 1.21 cm, EPSG:6341 |
| `04_qa_summary.csv` | `artifact_mask_share_pct` | 0.39 |
| `12_threshold_values.csv` | `otsu_ndvi`, `otsu_redness` | 0.2514, 0.3804 |
| `11_reference_label_summary.csv` | test points | 2,000 / 2,000 / 2,000 / 452 |
| `15_accuracy_summary.csv` | overall accuracy, random forest | 0.9647 |
| `15_accuracy_summary.csv` | overall accuracy, index thresholds | 0.8442 |
| `21_site_cover.csv` | bare ground, random forest | 58.17% |
| `22_field_check_blm_aim.csv` | bare ground in the 30 m circle, random forest | 57.76% (field 53.33%) |
| `27_scale_comparison_statistics.csv` | green cover against Sentinel-2 NDVI, mean of both dates | r = 0.4195, n = 36 |

Random seeds are fixed in `scripts/common.py` (`SEED`). Small differences in the last digit can appear with other library versions, because the random forest and the resampling depend on them.

Then open the figures in `outputs/figures/` and look at:

- `01_overview_qa.png`: the tarps in the south-east corner and the six targets are covered by the blue mask.
- `02_brightness_banding.png`: panel (c) is flat near 1.0.
- `10_class_maps.png`: in the 6 m window, green and orange patches sit on the dark plants of the color image.

## 6. Open the maps in a GIS

`outputs/maps/` holds four GeoTIFFs in EPSG:6341 (NAD83(2011) / UTM zone 12N):

- `class_random_forest.tif` and `class_threshold.tif`: 1.2 cm class maps. Values: 1 green vegetation, 2 dry vegetation or litter, 3 bare ground, 4 shadow or other, 0 masked.
- `fractional_cover_1m_random_forest.tif` and `fractional_cover_1m_threshold.tif`: three bands in percent (green, dry, bare).

In QGIS or ArcGIS Pro, add the files and symbolize the class maps by category with the four classes. I did not run QGIS or ArcGIS for this project, so check these steps yourself before describing them as done.

## 7. Common problems

- **Download stops part way.** Run step 1 again. Finished files are skipped; a partial file has the ending `.part` and is replaced.
- **ScienceBase is slow or returns an error.** Try again later. The files can also be downloaded by hand from https://doi.org/10.5066/P9MMCJET (multispectral, natural color, lidar and ground control child items, site 7 files only) into `data/raw/`.
- **Memory error.** Close other programs. Step 3 and step 6 need the most memory.
- **The Sentinel-2 or BLM service has changed.** The small files in `data/sentinel2/` and `data/blm_aim_plot_USGS_NEON2.csv` are already in the repository, so the analysis still runs. Step 1 only fetches them when they are missing.
- **`laspy` cannot read the file.** The lidar file is plain `.las`, so no extra backend is needed. If the size check in step 1 passed, reinstall `laspy`.
