# -*- coding: utf-8 -*-
"""
Three-model comparison on the same dark-sky reference classification task
with spatially independent site holdout.

Models
------
1. Random Forest (RF)
2. Gradient Boosting Decision Tree (GBDT)
3. XGBoost (XGB)

Validation design
-----------------
- Positive samples are grouped by complete DarkSky/IDSP protected areas.
- Negative samples are grouped by curated city ID/name when available.
- Negative samples without a city identifier are grouped into spatial blocks.
- The same protected area/city/spatial block is never allowed to appear in
  both training and testing sets.
- The same site split is applied to all years and all three models.
- Model comparison uses repeated stratified site holdout.
- XGBoost SHAP weights are calculated only on spatially held-out test sites.

Important
---------
The current manuscript uses three indicators: AOD, Cloud and NTL.
Therefore this script intentionally ignores PM2.5 and Precipitation even if
they are present in an older training table.

The older script generated some negative samples directly from low-quality
raster quantiles. If the paper now claims that negative cities were selected
from external literature/authoritative knowledge, set
EXCLUDE_RASTER_DERIVED_NEGATIVES = True and ensure that curated city samples
are included in the input table with a city identifier column.
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier
import xgboost as xgb


# =========================================================
# 0. Paths and parameters
# =========================================================

TRAINING_CSV = Path(
    r"F:\Global night\DarkSky_multi_source_negative_model_fast"
    r"\training_table_DarkSky_multi_source_negative.csv"
)

DARKSKY_SHP = Path(
    r"E:\dark sky\验证数据\dark place"
    r"\2024-005_Spinner-et-al_IDSP-20241\IDSP_2024.shp"
)

OUT_DIR = Path(
    r"F:\Global night\DarkSky_three_models_spatial_validation"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2012, 2024))

# Keep this consistent with the current Methods.
FEATURE_NAMES = ["AOD", "Cloud", "NTL"]

LABEL_COL = "label"
YEAR_COL = "year"
LON_COL = "Longitude"
LAT_COL = "Latitude"
SOURCE_COL = "source"

# Candidate columns that may contain an externally assigned city identifier.
CITY_ID_CANDIDATES = [
    "city_id", "city_name", "City", "CITY", "CITY_NAME",
    "metro_id", "metro_name", "site_name"
]

# Candidate protected-area name/ID fields.
DARKSKY_ID_CANDIDATES = [
    "site_id", "Site_ID", "ID", "id", "Name", "NAME",
    "site_name", "SITE_NAME", "DESIGNATIO"
]

# If True, rows whose source starts with "Raster_" are excluded.
# Keep this True when the manuscript states that negative-city labels came
# from external knowledge rather than the study rasters.
EXCLUDE_RASTER_DERIVED_NEGATIVES = False
RASTER_NEGATIVE_PREFIX = "Raster_"

# Fallback grouping for GAN or other negative samples without city IDs.
NEGATIVE_BLOCK_DEG = 2.0

# Prevent a few very large sites from dominating the model.
MAX_SAMPLES_PER_SITE_YEAR = 200

# Repeated spatial holdout.
N_REPEATS = 20
TEST_SITE_FRACTION = 0.25
RANDOM_SEED = 20260714

# Permutation importance on held-out data.
PERMUTATION_REPEATS = 5
PERMUTATION_MAX_SAMPLES = 3000

# SHAP for XGBoost on held-out data.
SHAP_SAMPLE_N = 3000

# Model-selection score:
# final = PERFORMANCE_WEIGHT * performance + STABILITY_WEIGHT * stability
PERFORMANCE_WEIGHT = 0.30
STABILITY_WEIGHT = 0.70


# =========================================================
# 1. Utility functions
# =========================================================

def require_columns(df: pd.DataFrame, columns, table_name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def choose_existing_column(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def spatial_block_id(lon: pd.Series, lat: pd.Series, block_deg: float) -> pd.Series:
    lon_bin = np.floor((lon.astype(float) + 180.0) / block_deg).astype("Int64")
    lat_bin = np.floor((lat.astype(float) + 90.0) / block_deg).astype("Int64")
    return "BLOCK_" + lon_bin.astype(str) + "_" + lat_bin.astype(str)


def cap_samples_per_site_year(df: pd.DataFrame, max_n: int, seed: int) -> pd.DataFrame:
    """Cap rows within each site-year without breaking site groups."""
    if max_n is None or max_n <= 0:
        return df.copy()

    rng = np.random.default_rng(seed)
    kept = []
    for (_, _), sub in df.groupby(["site_id", YEAR_COL], sort=False):
        if len(sub) <= max_n:
            kept.append(sub)
        else:
            rs = int(rng.integers(0, 2**31 - 1))
            kept.append(sub.sample(n=max_n, random_state=rs))
    return pd.concat(kept, ignore_index=True)


# =========================================================
# 2. Build protected-area/city/site groups
# =========================================================

def assign_darksky_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each positive point to a complete DarkSky protected-area group."""
    pos = df[df[LABEL_COL] == 1].copy()
    if pos.empty:
        return pos

    dark = gpd.read_file(DARKSKY_SHP)
    if dark.crs is None:
        raise ValueError("DarkSky shapefile has no CRS.")

    dark = dark[~dark.geometry.isna()].copy()
    dark = dark[dark.geometry.is_valid].copy()
    dark = dark.explode(index_parts=False).reset_index(drop=True)

    id_col = choose_existing_column(dark.columns, DARKSKY_ID_CANDIDATES)
    if id_col is None:
        dark["_dark_id"] = np.arange(len(dark)).astype(str)
    else:
        dark["_dark_id"] = dark[id_col].astype(str) + "__" + dark.index.astype(str)

    pts = gpd.GeoDataFrame(
        pos.copy(),
        geometry=gpd.points_from_xy(pos[LON_COL], pos[LAT_COL]),
        crs="EPSG:4326"
    ).to_crs(dark.crs)

    joined = gpd.sjoin(
        pts,
        dark[["_dark_id", "geometry"]],
        how="left",
        predicate="intersects"
    )
    joined = joined[~joined.index.duplicated(keep="first")].copy()

    unmatched = joined["_dark_id"].isna()
    unmatched_rate = float(unmatched.mean())
    if unmatched_rate > 0:
        print(
            f"Warning: {unmatched.sum()} positive rows "
            f"({unmatched_rate:.2%}) did not intersect a DarkSky polygon."
        )

    fallback = spatial_block_id(
        joined.loc[unmatched, LON_COL],
        joined.loc[unmatched, LAT_COL],
        block_deg=0.25
    )
    joined.loc[unmatched, "_dark_id"] = "UNMATCHED_" + fallback
    joined["site_id"] = "POS_DARKSKY::" + joined["_dark_id"].astype(str)

    return pd.DataFrame(
        joined.drop(columns=["geometry", "index_right"], errors="ignore")
    )


def assign_negative_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Use externally assigned city IDs when present; otherwise use spatial blocks."""
    neg = df[df[LABEL_COL] == 0].copy()
    if neg.empty:
        return neg

    city_col = choose_existing_column(neg.columns, CITY_ID_CANDIDATES)

    if city_col is not None:
        city_value = neg[city_col].astype(str).str.strip()
        valid_city = (
            neg[city_col].notna()
            & ~city_value.isin(["", "nan", "None", "NaN"])
        )
    else:
        city_value = pd.Series("", index=neg.index)
        valid_city = pd.Series(False, index=neg.index)

    neg["site_id"] = ""

    if valid_city.any():
        neg.loc[valid_city, "site_id"] = "NEG_CITY::" + city_value.loc[valid_city]

    missing_city = ~valid_city
    fallback = spatial_block_id(
        neg.loc[missing_city, LON_COL],
        neg.loc[missing_city, LAT_COL],
        NEGATIVE_BLOCK_DEG
    )

    if SOURCE_COL in neg.columns:
        source = neg.loc[missing_city, SOURCE_COL].astype(str)
    else:
        source = pd.Series("UNKNOWN", index=neg.loc[missing_city].index)

    neg.loc[missing_city, "site_id"] = (
        "NEG_SPATIAL::" + source + "::" + fallback
    )

    return neg


def load_and_prepare_data() -> pd.DataFrame:
    print("\n>>> Loading training table...")
    if not TRAINING_CSV.exists():
        raise FileNotFoundError(TRAINING_CSV)
    if not DARKSKY_SHP.exists():
        raise FileNotFoundError(DARKSKY_SHP)

    df = pd.read_csv(TRAINING_CSV, low_memory=False)

    require_columns(
        df,
        [LON_COL, LAT_COL, YEAR_COL, LABEL_COL] + FEATURE_NAMES,
        "training table"
    )

    numeric_cols = [LON_COL, LAT_COL, YEAR_COL, LABEL_COL] + FEATURE_NAMES
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=numeric_cols).copy()
    df[YEAR_COL] = df[YEAR_COL].astype(int)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df = df[df[YEAR_COL].isin(YEARS) & df[LABEL_COL].isin([0, 1])].copy()

    for f in FEATURE_NAMES:
        df = df[np.isfinite(df[f])].copy()
        df = df[(df[f] >= 0) & (df[f] <= 1)].copy()

    if EXCLUDE_RASTER_DERIVED_NEGATIVES and SOURCE_COL in df.columns:
        raster_mask = (
            (df[LABEL_COL] == 0)
            & df[SOURCE_COL].astype(str).str.startswith(
                RASTER_NEGATIVE_PREFIX,
                na=False
            )
        )
        print("Excluded raster-derived negative rows:", int(raster_mask.sum()))
        df = df[~raster_mask].copy()

    pos = assign_darksky_groups(df)
    neg = assign_negative_groups(df)
    data = pd.concat([pos, neg], ignore_index=True)

    data = cap_samples_per_site_year(
        data,
        max_n=MAX_SAMPLES_PER_SITE_YEAR,
        seed=RANDOM_SEED
    )

    group_labels = data.groupby("site_id")[LABEL_COL].nunique()
    bad_groups = group_labels[group_labels > 1]
    if not bad_groups.empty:
        raise ValueError(
            "Some site groups contain both labels. Examples:\n"
            + str(bad_groups.head())
        )

    site_table = (
        data.groupby("site_id", as_index=False)
        .agg(
            label=(LABEL_COL, "first"),
            n_rows=(LABEL_COL, "size"),
            n_years=(YEAR_COL, "nunique")
        )
    )

    n_pos_sites = int((site_table["label"] == 1).sum())
    n_neg_sites = int((site_table["label"] == 0).sum())

    print("\nIndependent-site counts")
    print("-----------------------")
    print("Positive protected-area groups:", n_pos_sites)
    print("Negative city/spatial groups:", n_neg_sites)

    if min(n_pos_sites, n_neg_sites) < 4:
        raise ValueError(
            "Too few independent sites in one class. "
            "At least four positive and four negative groups are recommended."
        )

    prepared_csv = OUT_DIR / "training_table_with_site_groups.csv"
    site_csv = OUT_DIR / "site_group_summary.csv"
    data.to_csv(prepared_csv, index=False, encoding="utf-8-sig")
    site_table.to_csv(site_csv, index=False, encoding="utf-8-sig")

    print("Prepared table:", prepared_csv)
    print("Site summary:", site_csv)
    return data


# =========================================================
# 3. Candidate models
# =========================================================

def build_models(seed: int):
    return {
        "RF": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            max_features="sqrt",
            min_samples_leaf=2,
            bootstrap=True,
            n_jobs=-1,
            random_state=seed
        ),
        "GBDT": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=5,
            subsample=0.80,
            random_state=seed
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.80,
            colsample_bytree=0.80,
            min_child_weight=5,
            gamma=0.10,
            reg_lambda=2.0,
            reg_alpha=0.20,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed
        ),
    }


# =========================================================
# 4. Metrics and importance
# =========================================================

def safe_binary_metrics(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    classes = np.unique(y_true)

    if len(classes) == 2:
        roc_auc = roc_auc_score(y_true, prob)
        ap = average_precision_score(y_true, prob)
        ll = log_loss(y_true, prob, labels=[0, 1])
    else:
        roc_auc = np.nan
        ap = np.nan
        ll = np.nan

    cm = confusion_matrix(y_true, pred, labels=[0, 1])

    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc,
        "average_precision": ap,
        "brier": brier_score_loss(y_true, prob),
        "log_loss": ll,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def normalized_permutation_importance(model, X_test, y_test, seed):
    if len(X_test) > PERMUTATION_MAX_SAMPLES:
        X_eval = X_test.sample(n=PERMUTATION_MAX_SAMPLES, random_state=seed)
        y_eval = y_test.loc[X_eval.index]
    else:
        X_eval = X_test
        y_eval = y_test

    result = permutation_importance(
        model,
        X_eval,
        y_eval,
        scoring="roc_auc",
        n_repeats=PERMUTATION_REPEATS,
        random_state=seed,
        n_jobs=-1
    )

    raw = np.asarray(result.importances_mean, dtype=float)
    positive = np.clip(raw, 0.0, None)
    if positive.sum() <= 0:
        positive = np.abs(raw)

    if positive.sum() <= 0:
        normalized = np.full_like(positive, np.nan)
    else:
        normalized = positive / positive.sum()

    return raw, normalized


def get_xgb_shap_values_binary(model, X):
    """TreeSHAP values for the positive class using pred_contribs."""
    X2 = X.astype("float32").copy()
    booster = model.get_booster()

    dmat = xgb.DMatrix(X2, feature_names=list(X2.columns))
    contrib = booster.predict(
        dmat,
        pred_contribs=True,
        validate_features=False
    )
    contrib = np.asarray(contrib)

    if contrib.ndim == 2:
        shap_values = contrib[:, :-1]
    elif contrib.ndim == 3:
        if contrib.shape[1] == 1:
            shap_values = contrib[:, 0, :-1]
        else:
            shap_values = contrib[:, 1, :-1]
    else:
        raise ValueError(f"Unexpected XGBoost SHAP shape: {contrib.shape}")

    return shap_values


# =========================================================
# 5. Repeated spatially independent validation
# =========================================================

def run_spatial_validation(data: pd.DataFrame):
    site_table = (
        data.groupby("site_id", as_index=False)
        .agg(label=(LABEL_COL, "first"))
        .sort_values("site_id")
        .reset_index(drop=True)
    )

    splitter = StratifiedShuffleSplit(
        n_splits=N_REPEATS,
        test_size=TEST_SITE_FRACTION,
        random_state=RANDOM_SEED
    )

    metrics_rows = []
    importance_rows = []
    shap_rows = []
    split_rows = []

    X_sites = np.zeros((len(site_table), 1))
    y_sites = site_table["label"].to_numpy()

    for repeat, (train_idx, test_idx) in enumerate(
        splitter.split(X_sites, y_sites),
        start=1
    ):
        train_sites = set(site_table.loc[train_idx, "site_id"])
        test_sites = set(site_table.loc[test_idx, "site_id"])

        if train_sites & test_sites:
            raise RuntimeError("Site leakage detected.")

        for site_id in train_sites:
            split_rows.append({"repeat": repeat, "site_id": site_id, "subset": "train"})
        for site_id in test_sites:
            split_rows.append({"repeat": repeat, "site_id": site_id, "subset": "test"})

        print(
            f"\nRepeat {repeat}/{N_REPEATS}: "
            f"{len(train_sites)} train sites, {len(test_sites)} test sites"
        )

        for year in YEARS:
            train_df = data[
                (data[YEAR_COL] == year)
                & data["site_id"].isin(train_sites)
            ].copy()
            test_df = data[
                (data[YEAR_COL] == year)
                & data["site_id"].isin(test_sites)
            ].copy()

            if (
                train_df[LABEL_COL].nunique() < 2
                or test_df[LABEL_COL].nunique() < 2
            ):
                print(
                    f"Skipping year {year} in repeat {repeat}: "
                    "one class is absent."
                )
                continue

            X_train = train_df[FEATURE_NAMES].astype("float32")
            y_train = train_df[LABEL_COL].astype(int)
            X_test = test_df[FEATURE_NAMES].astype("float32")
            y_test = test_df[LABEL_COL].astype(int)

            sample_weight = compute_sample_weight(
                class_weight="balanced",
                y=y_train
            )

            models = build_models(seed=RANDOM_SEED + repeat * 100 + year)

            for model_name, base_model in models.items():
                model = clone(base_model)
                model.fit(X_train, y_train, sample_weight=sample_weight)

                prob = model.predict_proba(X_test)[:, 1]
                metric = safe_binary_metrics(y_test, prob)

                metrics_rows.append({
                    "repeat": repeat,
                    "year": year,
                    "model": model_name,
                    "n_train_rows": len(train_df),
                    "n_test_rows": len(test_df),
                    "n_train_sites": train_df["site_id"].nunique(),
                    "n_test_sites": test_df["site_id"].nunique(),
                    **metric
                })

                raw_imp, norm_imp = normalized_permutation_importance(
                    model,
                    X_test,
                    y_test,
                    seed=RANDOM_SEED + repeat * 1000 + year
                )

                for feature, raw_value, norm_value in zip(
                    FEATURE_NAMES,
                    raw_imp,
                    norm_imp
                ):
                    importance_rows.append({
                        "repeat": repeat,
                        "year": year,
                        "model": model_name,
                        "feature": feature,
                        "permutation_importance": raw_value,
                        "normalized_importance": norm_value
                    })

                if model_name == "XGBoost":
                    shap_n = min(SHAP_SAMPLE_N, len(X_test))
                    X_shap = X_test.sample(
                        n=shap_n,
                        random_state=RANDOM_SEED + repeat * 10000 + year
                    )

                    shap_values = get_xgb_shap_values_binary(model, X_shap)
                    mean_abs = np.abs(shap_values).mean(axis=0)
                    if mean_abs.sum() > 0:
                        normalized_shap = mean_abs / mean_abs.sum()
                    else:
                        normalized_shap = np.full_like(mean_abs, np.nan)

                    for feature, raw_value, norm_value in zip(
                        FEATURE_NAMES,
                        mean_abs,
                        normalized_shap
                    ):
                        shap_rows.append({
                            "repeat": repeat,
                            "year": year,
                            "feature": feature,
                            "mean_abs_shap": raw_value,
                            "normalized_shap_weight": norm_value
                        })

    metrics_df = pd.DataFrame(metrics_rows)
    importance_df = pd.DataFrame(importance_rows)
    shap_df = pd.DataFrame(shap_rows)
    split_df = pd.DataFrame(split_rows)

    if metrics_df.empty:
        raise ValueError(
            "No valid spatial validation folds were produced. "
            "Check the number and annual coverage of independent sites."
        )

    return metrics_df, importance_df, shap_df, split_df


# =========================================================
# 6. Model ranking
# =========================================================

def summarize_and_rank_models(metrics_df, importance_df):
    metric_summary = (
        metrics_df
        .groupby("model")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            average_precision_mean=("average_precision", "mean"),
            average_precision_std=("average_precision", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
            log_loss_mean=("log_loss", "mean"),
            n_evaluations=("model", "size")
        )
        .reset_index()
    )

    importance_summary = (
        importance_df
        .groupby(["model", "feature"])
        .agg(
            importance_mean=("normalized_importance", "mean"),
            importance_std=("normalized_importance", "std")
        )
        .reset_index()
    )

    eps = 1e-8
    importance_summary["importance_cv"] = (
        importance_summary["importance_std"]
        / (importance_summary["importance_mean"].abs() + eps)
    )
    importance_summary["feature_stability"] = (
        1.0 / (1.0 + importance_summary["importance_cv"])
    )

    stability_summary = (
        importance_summary
        .groupby("model")
        .agg(
            stability_score=("feature_stability", "mean"),
            max_importance_cv=("importance_cv", "max")
        )
        .reset_index()
    )

    ranking = metric_summary.merge(stability_summary, on="model", how="left")

    ranking["performance_score"] = ranking[
        [
            "roc_auc_mean",
            "average_precision_mean",
            "balanced_accuracy_mean",
            "f1_mean"
        ]
    ].mean(axis=1)

    ranking["performance_score"] = (
        ranking["performance_score"] * 4.0
        + (1.0 - ranking["brier_mean"])
    ) / 5.0

    ranking["final_score"] = (
        PERFORMANCE_WEIGHT * ranking["performance_score"]
        + STABILITY_WEIGHT * ranking["stability_score"]
    )

    ranking = ranking.sort_values("final_score", ascending=False).reset_index(drop=True)
    return metric_summary, importance_summary, stability_summary, ranking


def summarize_xgb_shap(shap_df):
    if shap_df.empty:
        return pd.DataFrame()

    summary = (
        shap_df
        .groupby("feature")
        .agg(
            mean_abs_shap_mean=("mean_abs_shap", "mean"),
            mean_abs_shap_std=("mean_abs_shap", "std"),
            normalized_weight_mean=("normalized_shap_weight", "mean"),
            normalized_weight_std=("normalized_shap_weight", "std"),
            n_estimates=("normalized_shap_weight", "size")
        )
        .reset_index()
    )

    eps = 1e-8
    summary["weight_cv"] = (
        summary["normalized_weight_std"]
        / (summary["normalized_weight_mean"].abs() + eps)
    )
    summary["weight_stability"] = 1.0 / (1.0 + summary["weight_cv"])

    return summary.sort_values(
        "normalized_weight_mean",
        ascending=False
    ).reset_index(drop=True)


# =========================================================
# 7. Figures
# =========================================================

def plot_auc(metrics_df):
    order = ["RF", "GBDT", "XGBoost"]
    values = [
        metrics_df.loc[metrics_df["model"] == m, "roc_auc"].dropna().values
        for m in order
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    ax.boxplot(values, labels=order, showmeans=True)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Spatially held-out ROC-AUC")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    path = OUT_DIR / "three_model_spatial_auc.png"
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()
    print("Figure:", path)


def plot_xgb_shap(shap_summary):
    if shap_summary.empty:
        return

    plot_df = shap_summary.sort_values("normalized_weight_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    ax.barh(
        plot_df["feature"],
        plot_df["normalized_weight_mean"],
        xerr=plot_df["normalized_weight_std"],
        capsize=3
    )
    ax.set_xlabel("Normalized mean |SHAP value|")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    path = OUT_DIR / "xgboost_spatial_holdout_shap_weights.png"
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()
    print("Figure:", path)


# =========================================================
# 8. Main
# =========================================================

def main():
    data = load_and_prepare_data()

    print("\n>>> Running repeated protected-area/city spatial holdout...")
    metrics_df, importance_df, shap_df, split_df = run_spatial_validation(data)

    (
        metric_summary,
        importance_summary,
        stability_summary,
        ranking
    ) = summarize_and_rank_models(metrics_df, importance_df)

    xgb_shap_summary = summarize_xgb_shap(shap_df)

    metrics_df.to_csv(
        OUT_DIR / "model_metrics_long.csv",
        index=False,
        encoding="utf-8-sig"
    )
    metric_summary.to_csv(
        OUT_DIR / "model_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )
    importance_df.to_csv(
        OUT_DIR / "permutation_importance_long.csv",
        index=False,
        encoding="utf-8-sig"
    )
    importance_summary.to_csv(
        OUT_DIR / "permutation_importance_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )
    stability_summary.to_csv(
        OUT_DIR / "model_stability_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )
    ranking.to_csv(
        OUT_DIR / "model_final_ranking.csv",
        index=False,
        encoding="utf-8-sig"
    )
    split_df.to_csv(
        OUT_DIR / "site_split_assignments.csv",
        index=False,
        encoding="utf-8-sig"
    )
    shap_df.to_csv(
        OUT_DIR / "xgboost_heldout_shap_long.csv",
        index=False,
        encoding="utf-8-sig"
    )
    xgb_shap_summary.to_csv(
        OUT_DIR / "xgboost_heldout_shap_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    plot_auc(metrics_df)
    plot_xgb_shap(xgb_shap_summary)

    print("\n==============================")
    print("Model ranking")
    print("==============================")
    print(
        ranking[
            [
                "model",
                "performance_score",
                "stability_score",
                "final_score",
                "roc_auc_mean",
                "average_precision_mean",
                "balanced_accuracy_mean",
                "f1_mean",
                "brier_mean"
            ]
        ].to_string(index=False)
    )

    print("\n==============================")
    print("XGBoost held-out SHAP weights")
    print("==============================")
    print(xgb_shap_summary.to_string(index=False))

    print("\nAll outputs:", OUT_DIR)


if __name__ == "__main__":
    main()
