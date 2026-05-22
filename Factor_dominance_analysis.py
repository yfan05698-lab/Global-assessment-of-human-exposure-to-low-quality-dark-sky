import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_window, geometry_mask
from rasterio.windows import Window
from shapely.geometry import mapping
from tqdm import tqdm


# =========================================================
# 1. Parameters
# =========================================================

BASE_DIR = r"E:\darksky"
OUT_DIR = Path(os.path.join(BASE_DIR, "factor_dominance_analysis"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2023]
# YEARS = list(range(2012, 2024))

FEATURES = ["PM25", "AOD", "Cloud", "Precip", "NTL"]

WEIGHTS = {
    "PM25": 0.5041,
    "AOD": 0.2155,
    "Cloud": 0.1155,
    "Precip": 0.1095,
    "NTL": 0.0554
}

LOW_DKQ_THRESHOLD = 0.7843

POLLUTION_TH = 0.60
METEOROLOGY_TH = 0.60
LIGHT_INVOLVED_TH = 0.10

BLOCK_SIZE = 2048

COUNTRY_SHP = r"E:/darksky/boundries/country.shp"
COUNTRY_FIELD = "country"

URBAN_GPKG = r"E:\darksky\data\capital\urban_top10_per_country_ESRI54009.gpkg"
URBAN_LAYER = "urban_top10"


URBAN_NAME_FIELD = None
URBAN_COUNTRY_FIELD = None

DSI_TEMPLATE = os.path.join(BASE_DIR, "DSI_batch_by_year", "DSI_{year}.tif")

POP_TEMPLATE = (
    r"E:\darksky\basic_data\pop\total_{year}_aligned_to_DSI_500m_conserve.tif"
)


def get_feature_files(year):
    return {
        "PM25": os.path.join(BASE_DIR, f"aligned_{year}", f"PM25_aligned_to_cl_{year}.tif"),
        "AOD": os.path.join(BASE_DIR, f"aligned_{year}", f"AOD_aligned_to_cl_{year}.tif"),
        "Cloud": os.path.join(BASE_DIR, f"aligned_{year}", f"cl_normalized_minmax_v{year}.tif"),
        "Precip": os.path.join(BASE_DIR, f"aligned_{year}", f"pre_normalized_minmax_v{year}.tif"),
        "NTL": os.path.join(BASE_DIR, f"aligned_{year}", f"NTL_aligned_to_cl_{year}.tif"),
    }


# =========================================================
# 2. Utility functions
# =========================================================

def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).replace("\ufeff", "").strip()


def valid_mask(arr, nodata):
    mask = np.isfinite(arr)
    if nodata is not None:
        if isinstance(nodata, float) and np.isnan(nodata):
            mask &= ~np.isnan(arr)
        else:
            mask &= arr != nodata
    return mask


def iter_windows(base_window, block_size):
    col0 = int(base_window.col_off)
    row0 = int(base_window.row_off)
    col1 = int(base_window.col_off + base_window.width)
    row1 = int(base_window.row_off + base_window.height)

    for row in range(row0, row1, block_size):
        for col in range(col0, col1, block_size):
            yield Window(
                col,
                row,
                min(block_size, col1 - col),
                min(block_size, row1 - row)
            )


def normalize(values):
    values = np.asarray(values, dtype="float64")
    s = np.nansum(values)
    if not np.isfinite(s) or s <= 0:
        return np.full(len(values), np.nan)
    return values / s


def classify_type(row, prefix):
    pm25 = row.get(f"{prefix}_R_PM25", np.nan)
    aod = row.get(f"{prefix}_R_AOD", np.nan)
    cloud = row.get(f"{prefix}_R_Cloud", np.nan)
    precip = row.get(f"{prefix}_R_Precip", np.nan)

    if np.all(pd.isna([pm25, aod, cloud, precip])):
        return "No low-DKQ"

    pollution = np.nansum([pm25, aod])
    meteorology = np.nansum([cloud, precip])

    if pollution >= POLLUTION_TH:
        return "Pollution-dominated"
    if meteorology >= METEOROLOGY_TH:
        return "Meteorology-dominated"
    return "Mixed"


def top_factor(row, prefix):
    vals = {
        f: row.get(f"{prefix}_R_{f}", np.nan)
        for f in FEATURES
    }
    vals = {k: v for k, v in vals.items() if pd.notna(v)}
    if len(vals) == 0:
        return "undefined", np.nan
    k = max(vals, key=vals.get)
    return k, vals[k]


def light_involved(row, prefix):
    v = row.get(f"{prefix}_R_NTL", np.nan)
    return int(pd.notna(v) and v >= LIGHT_INVOLVED_TH)


def check_alignment(ref, other, name):
    if ref.width != other.width or ref.height != other.height:
        raise ValueError(f"{name} size does not match DSI.")
    if ref.crs != other.crs:
        raise ValueError(f"{name} CRS does not match DSI.")
    if not np.allclose(tuple(ref.transform), tuple(other.transform), rtol=0, atol=1e-8):
        raise ValueError(f"{name} transform does not match DSI.")


def auto_field(gdf, candidates):
    cols = {c.lower(): c for c in gdf.columns}
    for c in candidates:
        if c in gdf.columns:
            return c
        if c.lower() in cols:
            return cols[c.lower()]
    return None


# =========================================================
# 3. Prepare vector data
# =========================================================

def load_country():
    gdf = gpd.read_file(COUNTRY_SHP)
    gdf[COUNTRY_FIELD] = gdf[COUNTRY_FIELD].apply(clean_text)
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
    gdf = gdf.dissolve(by=COUNTRY_FIELD, as_index=False)

    gdf["region_type"] = "country"
    gdf["region_name"] = gdf[COUNTRY_FIELD]
    gdf["country"] = gdf[COUNTRY_FIELD]
    gdf["region_id"] = np.arange(1, len(gdf) + 1)
    return gdf[["region_type", "region_id", "region_name", "country", "geometry"]]


def load_urban():
    gdf = gpd.read_file(URBAN_GPKG, layer=URBAN_LAYER)
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()

    name_field = URBAN_NAME_FIELD or auto_field(
        gdf, ["urban_name", "city", "name", "NAME", "City", "capital"]
    )
    country_field = URBAN_COUNTRY_FIELD or auto_field(
        gdf, ["country", "Country", "COUNTRY", "NAME_0", "admin", "ADMIN"]
    )

    if name_field is None:
        gdf["region_name"] = [f"urban_{i+1}" for i in range(len(gdf))]
    else:
        gdf["region_name"] = gdf[name_field].apply(clean_text)

    if country_field is None:
        gdf["country"] = ""
    else:
        gdf["country"] = gdf[country_field].apply(clean_text)

    gdf["region_type"] = "urban_top10"
    gdf["region_id"] = np.arange(1, len(gdf) + 1)

    print("Urban name field:", name_field)
    print("Urban country field:", country_field)

    return gdf[["region_type", "region_id", "region_name", "country", "geometry"]]


# =========================================================
# 4. Core calculation
# =========================================================

def calculate_region(row, year, dsi_ds, factor_dss, pop_ds):
    geom = row.geometry

    result = {
        "year": year,
        "region_type": row["region_type"],
        "region_id": row["region_id"],
        "region_name": clean_text(row["region_name"]),
        "country": clean_text(row["country"])
    }

    try:
        base_win = geometry_window(dsi_ds, [mapping(geom)], pad_x=0, pad_y=0)
    except Exception:
        return empty_result(result)

    area_raw = {f: 0.0 for f in FEATURES}
    pop_raw = {f: 0.0 for f in FEATURES}
    pressure_sum = {f: 0.0 for f in FEATURES}
    pressure_count = {f: 0 for f in FEATURES}

    low_pixels = 0
    low_pop = 0.0
    pixel_area_km2 = abs(dsi_ds.transform.a * dsi_ds.transform.e) / 1e6

    for win in iter_windows(base_win, BLOCK_SIZE):
        dsi = dsi_ds.read(1, window=win).astype("float32")
        transform = dsi_ds.window_transform(win)

        geom_mask = geometry_mask(
            [mapping(geom)],
            out_shape=dsi.shape,
            transform=transform,
            invert=True
        )

        low_mask = (
            geom_mask
            & valid_mask(dsi, dsi_ds.nodata)
            & (dsi < LOW_DKQ_THRESHOLD)
        )

        if not np.any(low_mask):
            continue

        pop = pop_ds.read(1, window=win).astype("float64")
        low_mask &= valid_mask(pop, pop_ds.nodata) & (pop >= 0)

        if not np.any(low_mask):
            continue

        factors = {}
        for f in FEATURES:
            arr = factor_dss[f].read(1, window=win).astype("float32")
            factors[f] = arr
            low_mask &= valid_mask(arr, factor_dss[f].nodata)

        if not np.any(low_mask):
            continue

        low_pixels += int(np.sum(low_mask))
        low_pop += float(np.nansum(pop[low_mask]))

        for f in FEATURES:
            pressure = 1.0 - factors[f][low_mask].astype("float64")
            pressure = np.clip(pressure, 0.0, 1.0)

            contrib = WEIGHTS[f] * pressure
            area_raw[f] += float(np.nansum(contrib))
            pop_raw[f] += float(np.nansum(contrib * pop[low_mask]))

            pressure_sum[f] += float(np.nansum(pressure))
            pressure_count[f] += int(np.sum(np.isfinite(pressure)))

    result["low_pixel_count"] = low_pixels
    result["low_area_km2"] = low_pixels * pixel_area_km2
    result["low_population"] = low_pop

    if low_pixels == 0:
        return empty_result(result)

    area_share = normalize([area_raw[f] for f in FEATURES])
    pop_share = normalize([pop_raw[f] for f in FEATURES])

    for i, f in enumerate(FEATURES):
        result[f"area_R_{f}"] = area_share[i]
        result[f"pop_R_{f}"] = pop_share[i]
        result[f"mean_pressure_{f}"] = (
            pressure_sum[f] / pressure_count[f]
            if pressure_count[f] > 0 else np.nan
        )

    result = add_type_fields(result)
    return result


def empty_result(result):
    result["low_pixel_count"] = 0
    result["low_area_km2"] = 0.0
    result["low_population"] = 0.0

    for f in FEATURES:
        result[f"area_R_{f}"] = np.nan
        result[f"pop_R_{f}"] = np.nan
        result[f"mean_pressure_{f}"] = np.nan

    result = add_type_fields(result)
    return result


def add_type_fields(result):
    s = pd.Series(result)

    result["area_type"] = classify_type(s, "area")
    result["pop_type"] = classify_type(s, "pop")

    result["area_pollution_share"] = np.nansum([
        result.get("area_R_PM25", np.nan),
        result.get("area_R_AOD", np.nan)
    ])
    result["area_meteorology_share"] = np.nansum([
        result.get("area_R_Cloud", np.nan),
        result.get("area_R_Precip", np.nan)
    ])
    result["pop_pollution_share"] = np.nansum([
        result.get("pop_R_PM25", np.nan),
        result.get("pop_R_AOD", np.nan)
    ])
    result["pop_meteorology_share"] = np.nansum([
        result.get("pop_R_Cloud", np.nan),
        result.get("pop_R_Precip", np.nan)
    ])

    result["area_light_involved"] = light_involved(s, "area")
    result["pop_light_involved"] = light_involved(s, "pop")

    area_top, area_val = top_factor(s, "area")
    pop_top, pop_val = top_factor(s, "pop")

    result["area_top_factor"] = area_top
    result["area_top_value"] = area_val
    result["pop_top_factor"] = pop_top
    result["pop_top_value"] = pop_val

    return result


def run_for_regions(gdf, year, dsi_ds, factor_dss, pop_ds, tag):
    if gdf.crs != dsi_ds.crs:
        gdf = gdf.to_crs(dsi_ds.crs)

    results = []
    iterator = tqdm(gdf.iterrows(), total=len(gdf), desc=f"{year} {tag}")

    for _, row in iterator:
        iterator.set_postfix({"region": clean_text(row["region_name"])[:20]})
        results.append(calculate_region(row, year, dsi_ds, factor_dss, pop_ds))

    df = pd.DataFrame(results)
    out_csv = OUT_DIR / f"{tag}_factor_dominance_{year}.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved:", out_csv)

    summary = (
        df.groupby(["area_type"], dropna=False)
        .agg(
            n_region=("region_id", "count"),
            total_low_area_km2=("low_area_km2", "sum"),
            total_low_population=("low_population", "sum"),
            mean_pollution_share=("area_pollution_share", "mean"),
            mean_meteorology_share=("area_meteorology_share", "mean")
        )
        .reset_index()
    )

    summary_csv = OUT_DIR / f"{tag}_factor_dominance_summary_{year}.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print("Saved:", summary_csv)

    return df


# =========================================================
# 5. Main
# =========================================================

def main():
    country_gdf = load_country()
    urban_gdf = load_urban()

    for year in YEARS:
        print("\n" + "=" * 80)
        print("Year:", year)
        print("=" * 80)

        dsi_file = DSI_TEMPLATE.format(year=year)
        pop_file = POP_TEMPLATE.format(year=year)
        factor_files = get_feature_files(year)

        for fp in [dsi_file, pop_file] + list(factor_files.values()):
            if not os.path.exists(fp):
                raise FileNotFoundError(fp)

        with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS", GDAL_CACHEMAX=4096):
            with rasterio.open(dsi_file) as dsi_ds, rasterio.open(pop_file) as pop_ds:
                factor_dss = {f: rasterio.open(factor_files[f]) for f in FEATURES}

                try:
                    check_alignment(dsi_ds, pop_ds, "Population")
                    for f in FEATURES:
                        check_alignment(dsi_ds, factor_dss[f], f)

                    run_for_regions(
                        country_gdf,
                        year,
                        dsi_ds,
                        factor_dss,
                        pop_ds,
                        tag="country"
                    )

                    run_for_regions(
                        urban_gdf,
                        year,
                        dsi_ds,
                        factor_dss,
                        pop_ds,
                        tag="urban_top10"
                    )

                finally:
                    for ds in factor_dss.values():
                        ds.close()

    print("\nDone. Output directory:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()