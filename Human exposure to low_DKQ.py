import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from tqdm import tqdm


# =========================================================
# 1. Parameters
# =========================================================

YEARS = list(range(2012, 2024))

BASE_DIR = r"E:\darksky"

DSI_TEMPLATE = os.path.join(
    BASE_DIR,
    "DSI_batch_by_year",
    "DSI_{year}.tif"
)

# 总人口栅格
POP_TOTAL_TEMPLATE = (
    r"E:\darksky\basic_data\pop\total_{year}_aligned_to_DSI_500m_conserve.tif"
)

# 分年龄组人口栅格
AGE_GROUPS = ["0-14", "15-24", "25-44", "45-64", "65p"]

POP_AGE_TEMPLATE = (
    r"E:\darksky\basic_data\pop\total_{year}_{age}_aligned_to_DSI_500m_conserve.tif"
)

# 低质量暗夜天空阈值
LOW_DKQ_THRESHOLD = 0.7843

# 分块大小
WINDOW_SIZE = 2048

# 输出目录
OUT_DIR = Path(os.path.join(BASE_DIR, "population_exposure"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. Utility functions
# =========================================================

def check_file(fp):
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)


def read_as_float(src, window):
    arr = src.read(1, window=window).astype("float64")

    if src.nodata is not None:
        try:
            if np.isnan(src.nodata):
                arr[np.isnan(arr)] = np.nan
            else:
                arr[arr == src.nodata] = np.nan
        except TypeError:
            arr[arr == src.nodata] = np.nan

    arr[np.isinf(arr)] = np.nan
    arr[arr < -1e20] = np.nan

    return arr


def valid_mask(arr):
    return np.isfinite(arr)


def iter_windows(width, height, window_size):
    for row in range(0, height, window_size):
        win_h = min(window_size, height - row)

        for col in range(0, width, window_size):
            win_w = min(window_size, width - col)
            yield Window(col, row, win_w, win_h)


def check_alignment(ref_ds, other_ds, name):
    if ref_ds.width != other_ds.width or ref_ds.height != other_ds.height:
        raise ValueError(
            f"{name} 与 DSI 行列数不一致：\n"
            f"{name}: {other_ds.width} x {other_ds.height}\n"
            f"DSI: {ref_ds.width} x {ref_ds.height}"
        )

    if ref_ds.crs != other_ds.crs:
        raise ValueError(
            f"{name} 与 DSI CRS 不一致：\n"
            f"{name}: {other_ds.crs}\n"
            f"DSI: {ref_ds.crs}"
        )

    if not np.allclose(
        tuple(ref_ds.transform),
        tuple(other_ds.transform),
        rtol=0,
        atol=1e-8
    ):
        raise ValueError(
            f"{name} 与 DSI transform 不一致：\n"
            f"{name}: {other_ds.transform}\n"
            f"DSI: {ref_ds.transform}"
        )


def pixel_area_km2(src):
    return abs(src.transform.a * src.transform.e) / 1e6


# =========================================================
# 3. Annual exposure calculation
# =========================================================

def calculate_exposure_for_year(year):
    print("\n" + "=" * 80)
    print(f"Processing year: {year}")
    print("=" * 80)

    dsi_file = DSI_TEMPLATE.format(year=year)
    pop_total_file = POP_TOTAL_TEMPLATE.format(year=year)

    check_file(dsi_file)
    check_file(pop_total_file)

    age_files = {
        age: POP_AGE_TEMPLATE.format(year=year, age=age)
        for age in AGE_GROUPS
    }

    for age, fp in age_files.items():
        check_file(fp)

    annual_total_result = {}
    annual_age_results = []

    with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS", GDAL_CACHEMAX=4096):
        with rasterio.open(dsi_file) as dsi_ds, rasterio.open(pop_total_file) as pop_total_ds:
            check_alignment(dsi_ds, pop_total_ds, "Total population")

            age_dss = {
                age: rasterio.open(age_files[age])
                for age in AGE_GROUPS
            }

            try:
                for age, ds in age_dss.items():
                    check_alignment(dsi_ds, ds, f"Age population {age}")

                px_area = pixel_area_km2(dsi_ds)

                low_pixel_count = 0
                low_area_km2 = 0.0
                exposed_total_pop = 0.0

                exposed_age_pop = {
                    age: 0.0 for age in AGE_GROUPS
                }

                iterator = tqdm(
                    list(iter_windows(dsi_ds.width, dsi_ds.height, WINDOW_SIZE)),
                    desc=f"[{year}] low-DKQ exposure",
                    ncols=100
                )

                for window in iterator:
                    dsi = read_as_float(dsi_ds, window)

                    low_mask = (
                        valid_mask(dsi)
                        & (dsi < LOW_DKQ_THRESHOLD)
                    )

                    if not np.any(low_mask):
                        continue

                    pop_total = read_as_float(pop_total_ds, window)
                    total_valid = valid_mask(pop_total) & (pop_total >= 0)

                    low_total_mask = low_mask & total_valid

                    if np.any(low_total_mask):
                        low_pixel_count += int(np.sum(low_total_mask))
                        exposed_total_pop += float(np.nansum(pop_total[low_total_mask]))

                    for age in AGE_GROUPS:
                        pop_age = read_as_float(age_dss[age], window)
                        age_valid = valid_mask(pop_age) & (pop_age >= 0)

                        low_age_mask = low_mask & age_valid

                        if np.any(low_age_mask):
                            exposed_age_pop[age] += float(np.nansum(pop_age[low_age_mask]))

                low_area_km2 = low_pixel_count * px_area

                annual_total_result = {
                    "year": year,
                    "low_pixel_count": low_pixel_count,
                    "low_area_km2": low_area_km2,
                    "exposed_total_population": exposed_total_pop,
                    "exposure_density_person_per_km2": (
                        exposed_total_pop / low_area_km2
                        if low_area_km2 > 0 else np.nan
                    )
                }

                for age in AGE_GROUPS:
                    annual_age_results.append({
                        "year": year,
                        "age_group": age,
                        "exposed_population": exposed_age_pop[age]
                    })

            finally:
                for ds in age_dss.values():
                    ds.close()

    return annual_total_result, annual_age_results


# =========================================================
# 4. Main
# =========================================================

def main():
    annual_total_results = []
    annual_age_results_all = []

    for year in YEARS:
        total_result, age_results = calculate_exposure_for_year(year)

        annual_total_results.append(total_result)
        annual_age_results_all.extend(age_results)

    # -----------------------------------------------------
    # 4.1 Annual total exposure
    # -----------------------------------------------------

    df_total = pd.DataFrame(annual_total_results)

    out_total = OUT_DIR / "annual_lowDKQ_population_exposure.csv"
    df_total.to_csv(out_total, index=False, encoding="utf-8-sig")
    print("Saved:", out_total)

    # -----------------------------------------------------
    # 4.2 Annual age exposure, long format
    # -----------------------------------------------------

    df_age_long = pd.DataFrame(annual_age_results_all)

    out_age_long = OUT_DIR / "annual_lowDKQ_age_exposure_long.csv"
    df_age_long.to_csv(out_age_long, index=False, encoding="utf-8-sig")
    print("Saved:", out_age_long)

    # -----------------------------------------------------
    # 4.3 Annual age exposure, wide format
    # -----------------------------------------------------

    df_age_wide = (
        df_age_long
        .pivot(index="year", columns="age_group", values="exposed_population")
        .reset_index()
    )

    # 按 AGE_GROUPS 固定列顺序
    ordered_cols = ["year"] + AGE_GROUPS
    df_age_wide = df_age_wide[ordered_cols]

    df_age_wide["all_age_groups_sum"] = df_age_wide[AGE_GROUPS].sum(axis=1)

    out_age_wide = OUT_DIR / "annual_lowDKQ_age_exposure_wide.csv"
    df_age_wide.to_csv(out_age_wide, index=False, encoding="utf-8-sig")
    print("Saved:", out_age_wide)

    # -----------------------------------------------------
    # 4.4 Cumulative total exposure
    # -----------------------------------------------------

    cumulative_total = pd.DataFrame([{
        "start_year": YEARS[0],
        "end_year": YEARS[-1],
        "cumulative_exposed_total_population": df_total["exposed_total_population"].sum(),
        "mean_annual_exposed_total_population": df_total["exposed_total_population"].mean(),
        "cumulative_low_area_km2": df_total["low_area_km2"].sum(),
        "mean_annual_low_area_km2": df_total["low_area_km2"].mean()
    }])

    out_cum_total = OUT_DIR / f"cumulative_lowDKQ_population_exposure_{YEARS[0]}_{YEARS[-1]}.csv"
    cumulative_total.to_csv(out_cum_total, index=False, encoding="utf-8-sig")
    print("Saved:", out_cum_total)

    # -----------------------------------------------------
    # 4.5 Cumulative age exposure
    # -----------------------------------------------------

    df_age_cumulative = (
        df_age_long
        .groupby("age_group", as_index=False)
        .agg(
            cumulative_exposed_population=("exposed_population", "sum"),
            mean_annual_exposed_population=("exposed_population", "mean")
        )
    )

    df_age_cumulative["start_year"] = YEARS[0]
    df_age_cumulative["end_year"] = YEARS[-1]

    total_cum_age = df_age_cumulative["cumulative_exposed_population"].sum()

    df_age_cumulative["share_of_age_group_exposure"] = (
        df_age_cumulative["cumulative_exposed_population"] / total_cum_age
        if total_cum_age > 0 else np.nan
    )

    # 固定年龄组顺序
    age_order = {age: i for i, age in enumerate(AGE_GROUPS)}
    df_age_cumulative["age_order"] = df_age_cumulative["age_group"].map(age_order)
    df_age_cumulative = (
        df_age_cumulative
        .sort_values("age_order")
        .drop(columns=["age_order"])
    )

    out_cum_age = OUT_DIR / f"cumulative_lowDKQ_age_exposure_{YEARS[0]}_{YEARS[-1]}.csv"
    df_age_cumulative.to_csv(out_cum_age, index=False, encoding="utf-8-sig")
    print("Saved:", out_cum_age)

    # -----------------------------------------------------
    # 4.6 Simple consistency check
    # -----------------------------------------------------

    check_df = df_total[["year", "exposed_total_population"]].merge(
        df_age_wide[["year", "all_age_groups_sum"]],
        on="year",
        how="left"
    )

    check_df["age_sum_minus_total"] = (
        check_df["all_age_groups_sum"] - check_df["exposed_total_population"]
    )

    check_df["relative_difference"] = (
        check_df["age_sum_minus_total"] / check_df["exposed_total_population"]
    )

    out_check = OUT_DIR / "check_total_population_vs_age_sum.csv"
    check_df.to_csv(out_check, index=False, encoding="utf-8-sig")
    print("Saved:", out_check)

    print("\nAll done.")
    print("Output directory:", OUT_DIR)


if __name__ == "__main__":
    main()