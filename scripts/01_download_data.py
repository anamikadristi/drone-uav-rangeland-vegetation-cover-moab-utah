"""Step 1. Download all inputs. No account or key is needed.

(a) Drone data for Moab site 7 from USGS ScienceBase, about 1.2 GB.
    Scholl, V.M., Burgess, M.A., Bauer, M.A., Brady, L.R., and Thurau, R.G.,
    2024, USGS data release, https://doi.org/10.5066/P9MMCJET (public domain).
    I only take the files for site 7 (the grazed rangeland plot).
    Files that are already present with the right size are skipped.
(b) The BLM AIM field plot that sits inside the drone footprint (one record
    from the public BLM TerrADat feature service). Saved in data/.
(c) Two Sentinel-2 L2A scenes (6 days before and 4 days after the flight).
    Only a small window around the site is read from the cloud-optimized
    GeoTIFFs on AWS (Earth Search STAC). Saved in data/sentinel2/.

Parts (b) and (c) are small and are kept in the repository, so the analysis
can be rerun even if those services change.
"""
import sys
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
from common import RAW, DATA, USER_AGENT

AIM_URL = ("https://services1.arcgis.com/KbxwQRRfWyEYLgp4/arcgis/rest/services/"
           "BLM_Natl_AIM_TerrADat_Hub/FeatureServer/0/query")
STAC = "https://earth-search.aws.element84.com/v1/search"
SITE_LONLAT = (-109.4207, 38.2065)
S2_IDS = ["S2B_12SXH_20230429_0_L2A", "S2B_12SXH_20230509_0_L2A"]
# window in the Sentinel-2 CRS (WGS 84 / UTM zone 12N), about 300 m around the site
S2_WINDOW = (637940.0, 4229570.0, 638610.0, 4230250.0)

SB = "https://www.sciencebase.gov/catalog/item/{}?format=json"
ITEMS = {
    # ScienceBase child item id: file names wanted
    "64ef59e0d34e4b6d8ab72b8e": [            # multispectral products
        "2023-05-05_Moab_Site7_Multi_Ortho_5band.tif",
        "2023-05-05_Moab_Site7_Multi_Ortho_5band.hdr",
        "2023-05-05_Moab_Site7_Multi_DSM.tif",
        "2023-05_Moab_UAS_Multispectral_metadata.xml",
    ],
    "64ee43a8d34e4b6d8ab70f77": [            # natural color products
        "2023-05-05_Moab_Site7_RGB_Ortho_3band.tif",
        "2023-05-05_Moab_Site7_RGB_DSM.tif",
        "2023-05_Moab_UAS_NaturalColor_RGB_metadata.xml",
    ],
    "64ef6626d34e4b6d8ab72c45": [            # drone lidar point cloud
        "2023-05-05_Moab_Site7_Lidar_PointCloud.las",
        "2023-05_Moab_UAS_Lidar_metadata.xml",
    ],
    "65a70d1bd34e5af967a46842": [            # ground control points
        "2023-05_Moab_UT_UAS_GCPs.csv",
        "2023-05_Moab_UAS_GroundControl_metadata.xml",
    ],
    "64ee4030d34e4b6d8ab70f36": [            # parent item
        "DataDictionary_UAS-Moab-UT-May-2023.csv",
    ],
}


def get_aim(s):
    """One BLM AIM terrestrial plot (USGS_NEON2) measured on 2023-04-30."""
    out = DATA / "blm_aim_plot_USGS_NEON2.csv"
    if out.exists():
        print("have", out.name)
        return
    q = {"where": "PlotID='USGS_NEON2'", "outFields": "*", "outSR": 4326,
         "returnGeometry": "true", "f": "json"}
    js = s.get(AIM_URL, params=q, timeout=120).json()
    rows = []
    for f in js["features"]:
        a = dict(f["attributes"])
        a["lon_wgs84"], a["lat_wgs84"] = f["geometry"]["x"], f["geometry"]["y"]
        rows.append(a)
    df = pd.DataFrame(rows)
    for c in ("DateVisited", "DateEstablished"):
        df[c] = pd.to_datetime(df[c], unit="ms").dt.strftime("%Y-%m-%d")
    df = df.dropna(axis=1, how="all")
    df.T.rename_axis("field").to_csv(out, header=["value"])
    print("saved", out.name, len(df), "record")


def get_sentinel2(s):
    """Read blue, green, red, NIR and the scene classification for a small window."""
    outdir = DATA / "sentinel2"
    outdir.mkdir(exist_ok=True)
    if all((outdir / f"{i}_window.tif").exists() for i in S2_IDS):
        print("have Sentinel-2 windows")
        return
    js = s.post(STAC, json={"collections": ["sentinel-2-l2a"], "ids": S2_IDS},
                timeout=120).json()
    for feat in js["features"]:
        out = outdir / f"{feat['id']}_window.tif"
        if out.exists():
            print("have", out.name)
            continue
        props = feat["properties"]
        bands = []
        for key in ("blue", "green", "red", "nir", "scl"):
            with rasterio.open(feat["assets"][key]["href"]) as src:
                if key == "blue":
                    win = from_bounds(*S2_WINDOW, src.transform).round_offsets().round_lengths()
                    tr = src.window_transform(win)
                    crs = src.crs
                    shape = (int(win.height), int(win.width))
                if key == "scl":   # 20 m layer, put on the 10 m grid
                    arr = np.zeros(shape, "float32")
                    reproject(rasterio.band(src, 1), arr, dst_transform=tr,
                              dst_crs=crs, resampling=Resampling.nearest)
                else:
                    arr = src.read(1, window=win).astype("float32")
                    # digital number to reflectance; newer baselines carry an offset
                    if not props.get("earthsearch:boa_offset_applied", False):
                        arr = arr - 1000.0
                    arr = arr / 10000.0
            bands.append(arr)
        prof = dict(driver="GTiff", height=shape[0], width=shape[1], count=5,
                    dtype="float32", crs=crs, transform=tr, compress="deflate")
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(np.stack(bands))
            dst.descriptions = ("B02 blue", "B03 green", "B04 red", "B08 nir", "SCL")
            dst.update_tags(scene_id=feat["id"], datetime=props["datetime"],
                            cloud_cover=str(props.get("eo:cloud_cover")),
                            processing_baseline=str(props.get("s2:processing_baseline")),
                            boa_offset_applied=str(props.get("earthsearch:boa_offset_applied")))
        print("saved", out.name, shape)


def main():
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    get_aim(s)
    get_sentinel2(s)
    for item, wanted in ITEMS.items():
        meta = s.get(SB.format(item), timeout=120).json()
        files = {f["name"]: f for f in meta["files"]}
        for name in wanted:
            f = files[name]
            out = RAW / name
            if out.exists() and out.stat().st_size == f["size"]:
                print("have", name)
                continue
            url = f.get("downloadUri") or f["url"]
            print("get ", name, round(f["size"] / 1e6, 1), "MB")
            tmp = out.with_suffix(out.suffix + ".part")
            with s.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            if tmp.stat().st_size != f["size"]:
                sys.exit(f"size mismatch for {name}")
            tmp.rename(out)
    print("done; raw files are in", RAW.name)


if __name__ == "__main__":
    main()
