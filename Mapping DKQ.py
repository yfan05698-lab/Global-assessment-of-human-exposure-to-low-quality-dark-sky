import os
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


# =========================================================
# 1. Parameters
# =========================================================

YEARS = list(range(2012, 2024))

BASE_DIR = r"E:\DarkSkyQuality"
ALIGNED_DIR_TEMPLATE = os.path.join(BASE_DIR, "aligned_{year}")
OUT_DIR = Path(os.path.join(BASE_DIR, "DSI_batch_by_year"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["PM25", "Cloud", "Precip", "NTL", "AOD"]

FILE_TEMPLATES = {
    "PM25": "PM25_aligned_to_cl_{year}.tif",
    "Cloud": "cl_normalized_minmax_v{year}.tif",
    "Precip": "pre_normalized_minmax_v{year}.tif",
    "NTL": "NTL_aligned_to_cl_{year}.tif",
    "AOD": "AOD_aligned_to_cl_{year}.tif",
}

# Raw SHAP-based weights.
RAW_WEIGHTS = {
    "PM25": 1.5480,
    "Cloud": 0.3548,
    "Precip": 0.3363,
    "NTL": 0.1702,
    "AOD": 0.6617,
}

# L1 normalization.
weight_sum = sum(abs(v) for v in RAW_WEIGHTS.values())
WEIGHTS = {k: float(v) / weight_sum for k, v in RAW_WEIGHTS.items()}

MIN_VALID_FRACTION = 1.0
WINDOW_SIZE = 1024
NODATA = np.nan


# =========================================================
# 2. Utility functions
# =========================================================

def get_feature_files(year):
    aligned_dir = ALIGNED_DIR_TEMPLATE.format(year=year)
    return {
        f: os.path.join(aligned_dir, FILE_TEMPLATES[f].format(year=year))
        for f in FEATURES
    }


def read_as_float(src, window):
    arr = src.read(1, window=window).astype("float32")

    if src.nodata is not None:
        arr[arr == src.nodata] = np.nan

    arr[np.isinf(arr)] = np.nan
    arr[arr < -1e20] = np.nan

    return arr


def check_alignment(ref_ds, other_ds, name):
    if ref_ds.width != other_ds.width or ref_ds.height != other_ds.height:
        raise ValueError(f"{name} does not match reference raster size.")

    if ref_ds.crs != other_ds.crs:
        raise ValueError(f"{name} does not match reference raster CRS.")

    if not np.allclose(
        tuple(ref_ds.transform),
        tuple(other_ds.transform),
        rtol=0,
        atol=1e-8
    ):
        raise ValueError(f"{name} does not match reference raster transform.")


def iter_windows(width, height, window_size):
    for row in range(0, height, window_size):
        win_h = min(window_size, height - row)

        for col in range(0, width, window_size):
            win_w = min(window_size, width - col)
            yield Window(col, row, win_w, win_h)


def get_valid_minmax(raster_path):
    with rasterio.open(raster_path) as src:
        gmin, gmax = np.inf, -np.inf

        for window in iter_windows(src.width, src.height, WINDOW_SIZE):
            block = read_as_float(src, window)

            if np.all(np.isnan(block)):
                continue

            gmin = min(gmin, float(np.nanmin(block)))
            gmax = max(gmax, float(np.nanmax(block)))

    if gmin == np.inf or gmax == -np.inf:
        return None, None

    return gmin, gmax


def normalize_raster_inplace(raster_path):
    gmin, gmax = get_valid_minmax(raster_path)

    if gmin is None:
        print(f"No valid pixels found in {raster_path}")
        return

    denom = gmax - gmin
    if denom == 0:
        denom = 1.0

    with rasterio.open(raster_path, "r+") as src:
        for window in iter_windows(src.width, src.height, WINDOW_SIZE):
            block = read_as_float(src, window)

            out = np.full(block.shape, np.nan, dtype="float32")
            valid = ~np.isnan(block)
            out[valid] = (block[valid] - gmin) / denom

            src.write(out, 1, window=window)


# =========================================================
# 3. Annual DSI mapping
# =========================================================

def calculate_dsi_for_year(year):
    print(f"\nProcessing year: {year}")

    feature_files = get_feature_files(year)

    for f, fp in feature_files.items():
        if not os.path.exists(fp):
            raise FileNotFoundError(f"{f} file not found: {fp}")

    out_path = OUT_DIR / f"DSI_{year}.tif"

    # Cloud is used as the reference grid.
    ref_file = feature_files["Cloud"]

    with rasterio.open(ref_file) as ref_ds:
        meta = ref_ds.meta.copy()
        meta.update(
            dtype="float32",
            nodata=NODATA,
            compress="lzw"
        )

        datasets = {
            f: rasterio.open(feature_files[f])
            for f in FEATURES
        }

        try:
            for f, ds in datasets.items():
                check_alignment(ref_ds, ds, f)

            with rasterio.open(out_path, "w", **meta) as dst:
                for window in iter_windows(ref_ds.width, ref_ds.height, WINDOW_SIZE):
                    arrays = {}

                    for f in FEATURES:
                        arrays[f] = read_as_float(datasets[f], window)

                    valid_count = np.zeros(arrays["Cloud"].shape, dtype="uint8")

                    for f in FEATURES:
                        valid_count += (~np.isnan(arrays[f])).astype("uint8")

                    required = math.ceil(MIN_VALID_FRACTION * len(FEATURES))
                    valid_pixels = valid_count >= required

                    dsi = np.zeros(arrays["Cloud"].shape, dtype="float32")

                    for f in FEATURES:
                        arr = np.nan_to_num(arrays[f], nan=0.0)
                        dsi += WEIGHTS[f] * arr

                    dsi[~valid_pixels] = np.nan

                    dst.write(dsi.astype("float32"), 1, window=window)

        finally:
            for ds in datasets.values():
                ds.close()

    normalize_raster_inplace(out_path)

    print(f"Saved annual DSI: {out_path}")

    return str(out_path)


# =========================================================
# 4. Multi-year mean DSI
# =========================================================

def calculate_mean_dsi(year_files):
    if len(year_files) == 0:
        return None

    mean_out = OUT_DIR / f"DSI_mean_{YEARS[0]}_{YEARS[-1]}.tif"

    with rasterio.open(year_files[0]) as ref_ds:
        meta = ref_ds.meta.copy()
        meta.update(
            dtype="float32",
            nodata=NODATA,
            compress="lzw"
        )

        with rasterio.open(mean_out, "w", **meta) as dst:
            for window in iter_windows(ref_ds.width, ref_ds.height, WINDOW_SIZE):
                sum_block = np.zeros((int(window.height), int(window.width)), dtype="float64")
                count_block = np.zeros((int(window.height), int(window.width)), dtype="int16")

                for fp in year_files:
                    with rasterio.open(fp) as src:
                        block = read_as_float(src, window)
                        valid = ~np.isnan(block)

                        sum_block[valid] += block[valid]
                        count_block[valid] += 1

                mean_block = np.full(sum_block.shape, np.nan, dtype="float32")
                valid_mean = count_block > 0
                mean_block[valid_mean] = (
                    sum_block[valid_mean] / count_block[valid_mean]
                ).astype("float32")

                dst.write(mean_block, 1, window=window)

    print(f"Saved multi-year mean DSI: {mean_out}")

    return str(mean_out)


# =========================================================
# 5. Run
# =========================================================

def main():
    print("Normalized weights:")
    for k, v in WEIGHTS.items():
        print(f"  {k}: {v:.4f}")

    year_files = []

    for year in YEARS:
        out = calculate_dsi_for_year(year)
        year_files.append(out)

    calculate_mean_dsi(year_files)

    print("\nAll done.")


if __name__ == "__main__":
    main()