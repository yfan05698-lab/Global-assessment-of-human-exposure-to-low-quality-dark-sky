# -*- coding: utf-8 -*-#
"""
DarkSky 正样本 + 多源空间平衡弱负样本
仅使用 AOD、Cloud、NTL 三个因子
XGBoost + TreeSHAP 重新估计 DSQ 因子权重（优化提速与泛化版）
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from affine import Affine
from shapely.geometry import Point
from pyproj import Transformer
from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

from xgboost import XGBClassifier
import xgboost as xgb
import matplotlib.pyplot as plt


# =========================================================
# 0. 路径与参数
# =========================================================

YEARS = list(range(2012, 2024))
GAN_CSV = r"F:\Global night\processed\GlobeAtNight_2012_2023_clean.csv"
DARKSKY_SHP = r"F:\global_dark_sky_places_original_processed_and_results\before_2024_after\park_reserve\IDSP_need_park_and_reserve.shp"
OUT_DIR = r"F:\Global night\DarkSky_AOD_Cloud_NTL_model_fast"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_NAMES = ["AOD", "Cloud", "NTL"]
FEATURE_ORIENTATION = "quality_high_good"

LAT_MIN, LAT_MAX = -55, 59
LON_MIN, LON_MAX = -170, 170

USE_GAN_NEGATIVE = True
GAN_LM_THRESHOLD = 3
USE_TIME_FILTER = True
LOCAL_TIME_START = 20.0
LOCAL_TIME_END = 4.0
USE_CLOUD_FILTER = True
CLOUD_THRESHOLD = 25

NEG_TOTAL_PER_YEAR = 3000
NEG_GAN_MAX_PER_YEAR = 1000
NEG_RASTER_MAX_PER_SOURCE_PER_YEAR = 800
MAX_NEG_PER_GRID_PER_YEAR = 5
GRID_STRIDE = 50
POS_MAX_PER_YEAR = 3000

DO_TUNING = False
TUNING_N_ITER = 30
N_REPEATS = 20
SHAP_SAMPLE_N = 5000

XGB_PARAMS = dict(
    n_estimators=500,  # 配合早停，可以适当调大上限
    max_depth=3,
    learning_rate=0.03,
    subsample=0.80,
    colsample_bytree=0.80,
    min_child_weight=5,
    gamma=0.1,
    reg_lambda=2.0,
    reg_alpha=0.2,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1
)


# =========================================================
# 1. 年度因子路径
# =========================================================

def get_raster_paths(year):
    base_dir = rf"E:\暗夜天空质量\暗夜天空质量\aligned_{year}"
    return {
        "AOD": os.path.join(base_dir, f"AOD_aligned_to_cl_{year}.tif"),
        "Cloud": os.path.join(base_dir, f"cl_normalized_minmax_v{year}.tif"),
        "NTL": os.path.join(base_dir, f"NTL_aligned_to_cl_{year}.tif"),
    }


# =========================================================
# 2. 输入检查
# =========================================================

def check_inputs():
    print("\n>>> 检查输入文件...")
    missing = []
    if not os.path.exists(DARKSKY_SHP):
        missing.append(DARKSKY_SHP)
    if USE_GAN_NEGATIVE and not os.path.exists(GAN_CSV):
        missing.append(GAN_CSV)

    for year in YEARS:
        paths = get_raster_paths(year)
        for name in FEATURE_NAMES:
            if not os.path.exists(paths[name]):
                missing.append(paths[name])

    if missing:
        print("以下文件不存在：")
        for p in missing[:20]:
            print(p)
        if len(missing) > 20:
            print(f"...... 共缺失 {len(missing)} 个文件")
        raise FileNotFoundError("输入文件缺失，请检查路径。")
    print("所有输入文件检查通过。")


# =========================================================
# 3. 通用函数
# =========================================================

def filter_study_extent(df, lon_col="Longitude", lat_col="Latitude"):
    return df[
        (df[lat_col] >= LAT_MIN) & (df[lat_col] <= LAT_MAX) &
        (df[lon_col] >= LON_MIN) & (df[lon_col] <= LON_MAX)
    ].copy()


def spatial_balance_points(df, lon_col="Longitude", lat_col="Latitude", year_col="year", max_per_grid=5, max_total=None, seed=42):
    if len(df) == 0:
        return df.copy()

    tmp = df.copy()
    tmp["lon_bin"] = np.floor(tmp[lon_col]).astype(int)
    tmp["lat_bin"] = np.floor(tmp[lat_col]).astype(int)
    out_list = []

    for year, sub_y in tmp.groupby(year_col):
        balanced = (
            sub_y
            .groupby(["lon_bin", "lat_bin"], group_keys=False)
            .apply(lambda x: x.sample(n=min(len(x), max_per_grid), random_state=int(seed + int(year))))
        )
        if max_total is not None and len(balanced) > max_total:
            balanced = balanced.sample(n=max_total, random_state=int(seed + int(year)))
        out_list.append(balanced)

    if not out_list:
        return pd.DataFrame()

    out = pd.concat(out_list, ignore_index=True)
    return out.drop(columns=["lon_bin", "lat_bin"], errors="ignore")


def quality_low_mask(values, q=0.20):
    values = np.asarray(values, dtype="float32")
    if FEATURE_ORIENTATION == "quality_high_good":
        thr = np.nanquantile(values, q)
        mask = values <= thr
    elif FEATURE_ORIENTATION == "pressure_high_bad":
        thr = np.nanquantile(values, 1 - q)
        mask = values >= thr
    else:
        raise ValueError("FEATURE_ORIENTATION 只能是 quality_high_good 或 pressure_high_bad")
    return mask, thr


def get_reference_grid(paths):
    ref_path = paths["Cloud"]
    with rasterio.open(ref_path) as ref:
        info = {
            "path": ref_path,
            "crs": ref.crs,
            "transform": ref.transform,
            "width": ref.width,
            "height": ref.height
        }
    return info


# =========================================================
# 4. Globe at Night LM<=3 负样本
# =========================================================

def parse_year_from_gan(df):
    for c in ["year", "Year", "YEAR", "obs_year", "ObsYear"]:
        if c in df.columns:
            df["year"] = pd.to_numeric(df[c], errors="coerce")
            if df["year"].notna().sum() > 0:
                return df
    for c in ["LocalDate", "UTDate", "ObsDateTime", "Date", "date"]:
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors="coerce")
            if dt.notna().sum() > 0:
                df["year"] = dt.dt.year
                return df
    if "source_file" in df.columns:
        df["year"] = pd.to_numeric(df["source_file"].astype(str).str.extract(r"(20\d{2})")[0], errors="coerce")
        return df
    raise ValueError("无法识别 Globe 年份字段。")


def parse_local_time_hour(df):
    if "LocalTime" not in df.columns:
        df["local_hour"] = np.nan
        return df

    t = df["LocalTime"].astype(str).str.strip().replace(["nan", "NaN", "None", ""], np.nan)
    parsed = pd.to_datetime(t, format="%H:%M", errors="coerce")
    
    mask_na = parsed.isna() & t.notna()
    if mask_na.any():
        parsed.loc[mask_na] = pd.to_datetime(t[mask_na], format="%H:%M:%S", errors="coerce")
    mask_na = parsed.isna() & t.notna()
    if mask_na.any():
        parsed.loc[mask_na] = pd.to_datetime(t[mask_na], errors="coerce")

    df["local_hour"] = parsed.dt.hour + parsed.dt.minute / 60.0 + parsed.dt.second / 3360.0
    return df


def read_gan_lm_negative_samples():
    if not USE_GAN_NEGATIVE:
        return pd.DataFrame()

    print("\n>>> 读取 Globe at Night LM<=3 弱负样本...")
    df = pd.read_csv(GAN_CSV, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    for c in ["Latitude", "Longitude", "LimitingMag"]:
        if c not in df.columns:
            raise ValueError(f"Globe 数据缺少字段: {c}")

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["LimitingMag"] = pd.to_numeric(df["LimitingMag"], errors="coerce")

    df = parse_year_from_gan(df)
    df = parse_local_time_hour(df)

    before = len(df)
    df = df.dropna(subset=["Latitude", "Longitude", "LimitingMag", "year"]).copy()
    df = df[(df["Latitude"] >= -90) & (df["Latitude"] <= 90) & (df["Longitude"] >= -180) & (df["Longitude"] <= 180)].copy()
    df = df[~((df["Latitude"] == 0) & (df["Longitude"] == 0))].copy()
    df = df[df["year"].isin(YEARS)].copy()
    df = filter_study_extent(df)

    print(f"Globe 基础筛选: {before} -> {len(df)}")

    if USE_TIME_FILTER:
        before_time = len(df)
        df = df.dropna(subset=["local_hour"]).copy()
        if LOCAL_TIME_START < LOCAL_TIME_END:
            df = df[(df["local_hour"] >= LOCAL_TIME_START) & (df["local_hour"] < LOCAL_TIME_END)].copy()
        else:
            df = df[(df["local_hour"] >= LOCAL_TIME_START) | (df["local_hour"] < LOCAL_TIME_END)].copy()
        print(f"Globe 时间筛选 {LOCAL_TIME_START}:00 到次日 {LOCAL_TIME_END}:00: {before_time} -> {len(df)}")

    before_lm = len(df)
    df = df[df["LimitingMag"] <= GAN_LM_THRESHOLD].copy()
    print(f"Globe LM <= {GAN_LM_THRESHOLD}: {before_lm} -> {len(df)}")

    if USE_CLOUD_FILTER and "CloudCover" in df.columns:
        before_cloud = len(df)
        df["CloudCover"] = pd.to_numeric(df["CloudCover"], errors="coerce")
        df = df[(df["CloudCover"].isna()) | (df["CloudCover"] < 0) | (df["CloudCover"] <= CLOUD_THRESHOLD)].copy()
        print(f"Globe CloudCover 筛选: {before_cloud} -> {len(df)}")

    df["label"] = 0
    df["source"] = "GAN_LM_le3"

    df = spatial_balance_points(df, max_per_grid=MAX_NEG_PER_GRID_PER_YEAR, max_total=NEG_GAN_MAX_PER_YEAR, seed=100)
    print(f"Globe LM<=3 平衡后负样本数: {len(df)}")
    
    keep_cols = ["Longitude", "Latitude", "year", "label", "source"]
    extra_cols = [c for c in ["LimitingMag", "LocalDate", "LocalTime", "Country"] if c in df.columns]
    return df[keep_cols + extra_cols].copy()


# =========================================================
# 5. 快速读取年度候选网格
# =========================================================

def read_downsampled_feature_stack(year):
    print(f"\n快速读取 {year} 年降采样候选网格...")
    paths = get_raster_paths(year)
    ref = get_reference_grid(paths)

    out_h = int(np.ceil(ref["height"] / GRID_STRIDE))
    out_w = int(np.ceil(ref["width"] / GRID_STRIDE))

    scale_x = ref["width"] / out_w
    scale_y = ref["height"] / out_h
    out_transform = ref["transform"] * Affine.scale(scale_x, scale_y)

    rows, cols = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing="ij")
    xs = out_transform.c + (cols + 0.5) * out_transform.a + (rows + 0.5) * out_transform.b
    ys = out_transform.f + (cols + 0.5) * out_transform.d + (rows + 0.5) * out_transform.e

    xs_flat = xs.ravel().astype("float64")
    ys_flat = ys.ravel().astype("float64")
    rows_flat = rows.ravel()
    cols_flat = cols.ravel()

    crs_str = str(ref["crs"]).upper()
    if ref["crs"] is not None and crs_str not in ["EPSG:4326", "OGC:CRS84"]:
        transformer = Transformer.from_crs(ref["crs"], "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(xs_flat, ys_flat, errcheck=False)
        lon = np.asarray(lon, dtype="float64")
        lat = np.asarray(lat, dtype="float64")
    else:
        lon = xs_flat.copy()
        lat = ys_flat.copy()

    valid_geo = np.isfinite(lon) & np.isfinite(lat) & (lon >= -180) & (lon <= 180) & (lat >= -90) & (lat <= 90)
    valid_extent = (lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon <= LON_MAX)
    final_mask = valid_geo & valid_extent

    if final_mask.sum() == 0:
        return pd.DataFrame()

    data = pd.DataFrame({
        "Longitude": lon[final_mask],
        "Latitude": lat[final_mask],
        "year": year,
        "grid_row": rows_flat[final_mask],
        "grid_col": cols_flat[final_mask]
    })

    for name, path in paths.items():
        with rasterio.open(path) as src:
            vrt = WarpedVRT(src, crs=ref["crs"], transform=ref["transform"], width=ref["width"], height=ref["height"],
                            resampling=Resampling.bilinear, src_nodata=src.nodata, nodata=-9999.0)
            arr = vrt.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype("float32")
            vrt.close()

        arr_flat = arr.ravel()
        vals = arr_flat[final_mask].astype("float32")
        vals[vals == -9999.0] = np.nan
        vals[~np.isfinite(vals)] = np.nan
        vals[(vals < -1e20) | (vals > 1e20)] = np.nan
        data[name] = vals

    data = data.dropna(subset=FEATURE_NAMES).copy()
    for f in FEATURE_NAMES:
        data = data[(data[f] >= 0) & (data[f] <= 1)].copy()
    return data


# =========================================================
# 6. 栅格多源弱负样本
# =========================================================

def build_raster_weak_negatives_for_year(year):
    pool = read_downsampled_feature_stack(year)
    if len(pool) == 0:
        return pd.DataFrame()

    neg_list = []
    # 1. 各因子低质量候选
    for f_name in FEATURE_NAMES:
        mask, thr = quality_low_mask(pool[f_name].values, q=0.20)
        sub_df = pool.loc[mask].copy()
        sub_df["source"] = f"Raster_{f_name}_low_quality_thr_{thr:.4f}"
        neg_list.append(sub_df)

    # 2. 综合低质量候选
    composite_score = pool[FEATURE_NAMES].mean(axis=1)
    if FEATURE_ORIENTATION == "quality_high_good":
        thr_comp = np.nanquantile(composite_score, 0.15)
        mask_comp = composite_score <= thr_comp
    else:
        thr_comp = np.nanquantile(composite_score, 0.85)
        mask_comp = composite_score >= thr_comp

    comp_df = pool.loc[mask_comp].copy()
    comp_df["source"] = f"Raster_Composite_low_quality_thr_{thr_comp:.4f}"
    neg_list.append(comp_df)

    selected = []
    for sub in neg_list:
        if len(sub) == 0:
            continue
        sub = sub.drop_duplicates(subset=["grid_row", "grid_col"]).copy()
        sub["label"] = 0
        sub = spatial_balance_points(sub, max_per_grid=MAX_NEG_PER_GRID_PER_YEAR, max_total=NEG_RASTER_MAX_PER_SOURCE_PER_YEAR, seed=7000 + year)
        selected.append(sub)

    if not selected:
        return pd.DataFrame()

    neg = pd.concat(selected, ignore_index=True).drop_duplicates(subset=["year", "grid_row", "grid_col"]).copy()
    neg = spatial_balance_points(neg, max_per_grid=MAX_NEG_PER_GRID_PER_YEAR, max_total=NEG_TOTAL_PER_YEAR, seed=9000 + year)
    
    keep = ["Longitude", "Latitude", "year", "label", "source"] + FEATURE_NAMES
    return neg[keep].copy()


def build_all_raster_weak_negatives():
    all_neg = []
    for year in YEARS:
        neg_y = build_raster_weak_negatives_for_year(year)
        if len(neg_y) > 0:
            all_neg.append(neg_y)
    if not all_neg:
        raise ValueError("没有构建到任何栅格弱负样本。")
    return pd.concat(all_neg, ignore_index=True)


def build_all_negative_samples():
    neg_parts = []
    gan_neg = read_gan_lm_negative_samples()
    if len(gan_neg) > 0:
        neg_parts.append(gan_neg)

    raster_neg = build_all_raster_weak_negatives()
    if len(raster_neg) > 0:
        neg_parts.append(raster_neg)

    neg = pd.concat(neg_parts, ignore_index=True)
    final_list = []
    for year, sub_y in neg.groupby("year"):
        sub_y = spatial_balance_points(sub_y, max_per_grid=MAX_NEG_PER_GRID_PER_YEAR, max_total=NEG_TOTAL_PER_YEAR, seed=12000 + int(year))
        final_list.append(sub_y)

    neg_final = pd.concat(final_list, ignore_index=True)
    out_csv = os.path.join(OUT_DIR, "negative_samples_spatial_balanced.csv")
    neg_final.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("弱负样本输出：", out_csv)
    return neg_final


# =========================================================
# 7. DarkSky 正样本
# =========================================================

def sample_points_in_polygons(gdf, n, seed=42, sample_crs="ESRI:54009"):
    rng = np.random.default_rng(seed)
    gdf_eq = gdf[~gdf.geometry.isna()].to_crs(sample_crs)
    gdf_eq["geometry"] = gdf_eq.geometry.buffer(0)
    gdf_eq = gdf_eq[gdf_eq.geometry.is_valid & (gdf_eq.geometry.area > 0)].explode(index_parts=False).reset_index(drop=True)

    if len(gdf_eq) == 0:
        raise ValueError("DarkSky shp 中没有有效面。")

    geoms = list(gdf_eq.geometry)
    areas = np.asarray([geom.area for geom in geoms], dtype="float64")
    probs = areas / areas.sum()

    points = []
    attempts, max_attempts = 0, n * 1000
    pbar = tqdm(total=n, desc="Sampling DarkSky positives")

    while len(points) < n and attempts < max_attempts:
        idx = rng.choice(len(geoms), p=probs)
        geom = geoms[idx]
        minx, miny, maxx, maxy = geom.bounds
        pt = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        attempts += 1
        if geom.contains(pt):
            points.append(pt)
            pbar.update(1)
    pbar.close()

    pts_eq = gpd.GeoDataFrame({"tmp": np.arange(len(points))}, geometry=points, crs=sample_crs)
    pts_wgs = pts_eq.to_crs("EPSG:4326")
    return pd.DataFrame({"Longitude": pts_wgs.geometry.x, "Latitude": pts_wgs.geometry.y})


def build_darksky_positive_samples(neg_df):
    print("\n>>> 构建 DarkSky 正样本...")
    dark = gpd.read_file(DARKSKY_SHP)
    dark = dark[~dark.geometry.isna() & dark.geometry.is_valid].copy()

    pos_list = []
    for year in YEARS:
        neg_n = int((neg_df["year"] == year).sum())
        n_this = min(POS_MAX_PER_YEAR, neg_n)
        if n_this <= 0:
            continue
        pts = sample_points_in_polygons(dark, n=n_this, seed=15000 + year)
        pts["year"] = year
        pts["label"] = 1
        pts["source"] = "DarkSky"
        pos_list.append(pts)

    pos = pd.concat(pos_list, ignore_index=True)
    out_csv = os.path.join(OUT_DIR, "positive_samples_DarkSky.csv")
    pos.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return pos


# =========================================================
# 8. ⚡ 优化提速后的特征采样函数 ⚡
# =========================================================

def sample_raster_values_for_year(points_df, year):
    sub = points_df[points_df["year"] == year].copy()
    if len(sub) == 0:
        return sub

    gdf = gpd.GeoDataFrame(sub, geometry=gpd.points_from_xy(sub["Longitude"], sub["Latitude"]), crs="EPSG:4326")
    raster_paths = get_raster_paths(year)

    for feature_name in FEATURE_NAMES:
        raster_path = raster_paths[feature_name]
        with rasterio.open(raster_path) as src:
            gdf_src = gdf.to_crs(src.crs)
            coords = [(geom.x, geom.y) for geom in gdf_src.geometry]
            
            # 【优化】使用 np.fromiter 直接将 C 级生成器转化为 Numpy 数组，规避普通 Python 列表扩容带来的性能损耗
            print(f" 提取 {feature_name} {year} 数据中...")
            gen = src.sample(coords)
            vals = np.fromiter((v[0] for v in gen), dtype=np.float32, count=len(coords))

            if src.nodata is not None:
                vals[vals == src.nodata] = np.nan
            vals[~np.isfinite(vals)] = np.nan
            vals[(vals < -1e20) | (vals > 1e20)] = np.nan
            sub[feature_name] = vals

    return sub.drop(columns="geometry", errors="ignore")


def extract_all_features(samples):
    print("\n>>> 为正负样本提取年度三因子值...")
    out = []
    for year in YEARS:
        sub = sample_raster_values_for_year(samples, year)
        if len(sub) > 0:
            out.append(sub)

    data = pd.concat(out, ignore_index=True).dropna(subset=FEATURE_NAMES + ["label"]).copy()
    for f in FEATURE_NAMES:
        data = data[(data[f] >= 0) & (data[f] <= 1)].copy()

    out_csv = os.path.join(OUT_DIR, "training_table_DarkSky.csv")
    data.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"训练表构建完成，有效样本总数: {len(data)}")
    return data


# =========================================================
# 9. 🤖 引入严格验证集早停机制的 XGBoost + TreeSHAP 🤖
# =========================================================

def tune_xgboost(data):
    print("\n>>> 开始 XGBoost 轻量调参...")
    pos, neg = data[data["label"] == 1], data[data["label"] == 0]
    n_each = min(len(pos), len(neg))
    df_tune = pd.concat([pos.sample(n=n_each, random_state=2024), neg.sample(n=n_each, random_state=2025)]).sample(frac=1, random_state=2026).reset_index(drop=True)

    X, y = df_tune[FEATURE_NAMES].astype("float32"), df_tune["label"].astype(int)
    param_dist = {
        "n_estimators": [300, 500],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.02, 0.03, 0.05],
        "min_child_weight": [3, 5, 8],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9]
    }

    base_model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist", n_jobs=1, random_state=42)
    search = RandomizedSearchCV(estimator=base_model, param_distributions=param_dist, n_iter=TUNING_N_ITER, scoring="roc_auc", cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), verbose=1, random_state=42, n_jobs=-1)
    search.fit(X, y)

    best_params = search.best_params_.copy()
    best_params.update({"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "n_jobs": -1})
    return best_params


def get_shap_values_binary(model, X):
    X2 = X.copy().astype("float32")
    dmat = xgb.DMatrix(X2, feature_names=list(X2.columns))
    contrib = np.asarray(model.get_booster().predict(dmat, pred_contribs=True, validate_features=False))
    return contrib[:, :-1] if contrib.ndim == 2 else (contrib[:, 0, :-1] if contrib.shape[1] == 1 else contrib[:, 1, :-1])


def train_repeated_xgb_with_shap(data, xgb_params):
    print("\n>>> 开始重复训练模型 (包含验证集早停机制 & TreeSHAP 重量计算)...")
    metrics_rows, report_rows, shap_rows, direction_rows = [], [], [], []

    for repeat in tqdm(range(N_REPEATS), desc="Repeated training"):
        pos, neg = data[data["label"] == 1], data[data["label"] == 0]
        n_each = min(len(pos), len(neg))
        df_all = pd.concat([pos.sample(n=n_each, random_state=1000 + repeat), neg.sample(n=n_each, random_state=2000 + repeat)]).sample(frac=1, random_state=3000 + repeat).reset_index(drop=True)

        X, y = df_all[FEATURE_NAMES].astype("float32"), df_all["label"].astype(int)
        
        # 1. 划分独立的测试集 (25%) 与大训练集 (75%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4000 + repeat, stratify=y)
        
        # 2. 【严谨改进】再从大训练集中切出 15% 做为独立验证集，专用于控制早停，杜绝把测试集喂给早停造成的潜在数据泄露
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=4500 + repeat, stratify=y_train)

        # 3. 将 early_stopping_rounds 声明于构造函数内（顺应新版 XGB 规范）
        model = XGBClassifier(
            **xgb_params,
            early_stopping_rounds=25, 
            random_state=5000 + repeat
        )
        
        # 4. 训练并结合早停验证
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        cm = confusion_matrix(y_test, pred)

        metrics_rows.append({
            "repeat": repeat, "n_train": len(X_tr), "n_val": len(X_val), "n_test": len(X_test),
            "accuracy": accuracy_score(y_test, pred), "f1": f1_score(y_test, pred), "auc": roc_auc_score(y_test, prob),
            "tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1]
        })

        rep = classification_report(y_test, pred, output_dict=True)
        for cls_name, vals in rep.items():
            if isinstance(vals, dict):
                row = {"repeat": repeat, "class": cls_name}
                row.update(vals)
                report_rows.append(row)

        # 5. TreeSHAP 特征贡献解算
        X_shap = X_test.sample(n=min(SHAP_SAMPLE_N, len(X_test)), random_state=6000 + repeat)
        sv = get_shap_values_binary(model, X_shap)
        mean_abs = np.abs(sv).mean(axis=0)
        norm_weight = mean_abs / (mean_abs.sum() + 1e-8)

        for fname, raw_v, norm_v in zip(FEATURE_NAMES, mean_abs, norm_weight):
            shap_rows.append({"repeat": repeat, "feature": fname, "mean_abs_shap": raw_v, "normalized_weight": norm_v})

        for fname in FEATURE_NAMES:
            pos_mean = df_all.loc[df_all["label"] == 1, fname].mean()
            neg_mean = df_all.loc[df_all["label"] == 0, fname].mean()
            direction_rows.append({
                "repeat": repeat, "feature": fname, "positive_DarkSky_mean": pos_mean,
                "negative_weak_mean": neg_mean, "negative_minus_positive": neg_mean - pos_mean
            })

    metrics_df = pd.DataFrame(metrics_rows)
    report_df = pd.DataFrame(report_rows)
    shap_df = pd.DataFrame(shap_rows)
    direction_df = pd.DataFrame(direction_rows)

    shap_summary = (
        shap_df.groupby("feature")
        .agg(mean_abs_shap_mean=("mean_abs_shap", "mean"), mean_abs_shap_std=("mean_abs_shap", "std"),
             normalized_weight_mean=("normalized_weight", "mean"), normalized_weight_std=("normalized_weight", "std"))
        .reset_index().sort_values("normalized_weight_mean", ascending=False)
    )

    direction_summary = direction_df.groupby("feature").agg(
        positive_DarkSky_mean=("positive_DarkSky_mean", "mean"),
        negative_weak_mean=("negative_weak_mean", "mean"),
        negative_minus_positive=("negative_minus_positive", "mean")
    ).reset_index()

    return metrics_df, report_df, shap_df, shap_summary.merge(direction_summary, on="feature", how="left"), direction_df


# =========================================================
# 10. 绘图
# =========================================================

def plot_shap_weights(shap_summary):
    plot_df = shap_summary.sort_values("normalized_weight_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.barh(plot_df["feature"], plot_df["normalized_weight_mean"], xerr=plot_df["normalized_weight_std"], capsize=3, color="skyblue", edgecolor="gray")
    ax.set_xlabel("Normalized Mean |SHAP value| (Weight Contribution)")
    ax.set_title("SHAP-based Feature Weights Re-estimation")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "SHAP_normalized_weights.png"), dpi=600, bbox_inches="tight")
    plt.close()


def plot_metrics(metrics_df):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.boxplot([metrics_df["accuracy"], metrics_df["f1"], metrics_df["auc"]], labels=["Accuracy", "F1-Score", "AUC"], showmeans=True)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metrics Score")
    ax.set_title("Robust Validation Matrix Across 20 Repeats")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "model_performance_repeated.png"), dpi=600, bbox_inches="tight")
    plt.close()


# =========================================================
# 11. 完整主流程入口（补齐因截断丢失的全部逻辑）
# =========================================================

def main():
    # 1. 基础检查
    check_inputs()

    # 2. 构建负样本与正样本
    neg_df = build_all_negative_samples()
    pos_df = build_darksky_positive_samples(neg_df)

    # 3. 合并正负样本空间点
    samples = pd.concat([pos_df, neg_df], ignore_index=True)
    samples_out = os.path.join(OUT_DIR, "combined_samples_coords.csv")
    samples.to_csv(samples_out, index=False, encoding="utf-8-sig")
    print(f"\n>>> 初始合并样本点保存至: {samples_out}")

    # 4. 提取多时相栅格三因子数据
    data = extract_all_features(samples)

    # 5. 超参数路由控制
    xgb_params = tune_xgboost(data) if DO_TUNING else XGB_PARAMS

    # 6. 多轮迭代稳健训练与权重逆向推导
    metrics_df, report_df, shap_df, shap_summary, direction_df = train_repeated_xgb_with_shap(data, xgb_params)

    # 7. 导出全套研究成果报表（使用 utf-8-sig 严防 Excel 打开乱码）
    metrics_df.to_csv(os.path.join(OUT_DIR, "xgb_metrics_20_repeats.csv"), index=False, encoding="utf-8-sig")
    report_df.to_csv(os.path.join(OUT_DIR, "xgb_classification_report.csv"), index=False, encoding="utf-8-sig")
    shap_df.to_csv(os.path.join(OUT_DIR, "xgb_shap_all_values.csv"), index=False, encoding="utf-8-sig")
    direction_df.to_csv(os.path.join(OUT_DIR, "xgb_feature_directions.csv"), index=False, encoding="utf-8-sig")
    
    # 重点：这个输出就是用于重新赋予 DSQ 指标权重的归一化价值参照
    shap_summary_out = os.path.join(OUT_DIR, "dsq_reestimated_weights_summary.csv")
    shap_summary.to_csv(shap_summary_out, index=False, encoding="utf-8-sig")

    print("\n=========================================================")
    print(">>> 权重重新估计全流程计算成功！核心权重成果单见以下文件：")
    print(f"    👉 {shap_summary_out}")
    print("=========================================================")

    # 8. 绘制并输出高分辨率研究配图
    print(">>> 正在渲染出版级科研制图...")
    plot_shap_weights(shap_summary)
    plot_metrics(metrics_df)
    print(f">>> 所有的统计配图已保存至文件夹: {OUT_DIR}")


if __name__ == "__main__":
    main()
