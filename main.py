import os
import json
import math
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Query, Header
from supabase import create_client

ROOT = Path(__file__).resolve().parent

FEATURE_SPEC = json.loads((ROOT / "model2_feature_spec.json").read_text())
PRODUCTION_CONFIG = json.loads((ROOT / "model2_production_config.json").read_text())
SELFTEST = json.loads((ROOT / "model_selftest.json").read_text())

FEATURE_COLUMNS = FEATURE_SPEC["feature_columns"]
BASELINE_DAYS = int(FEATURE_SPEC["baseline_days"])
BASELINE_EXCLUSION_HOURS = int(FEATURE_SPEC["baseline_exclusion_hours"])
BASELINE_EXCLUSION_DAYS = max(1, int(round(BASELINE_EXCLUSION_HOURS / 24)))
MIN_BASELINE_DAYS = int(FEATURE_SPEC["minimum_baseline_days"])
HORIZONS_HOURS = list(FEATURE_SPEC["history_horizons_hours"])

PUBLIC_SIGNALS = [
    "walking_prop",
    "resting_prop",
    "feeding_prop",
    "activity",
]

ROBUST_Z_CLIP = 12.0

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "").strip()
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "").strip()
FARM_TIMEZONE = os.getenv("FARM_TIMEZONE", "Africa/Nairobi").strip()
MODEL2_VERSION = os.getenv("MODEL2_VERSION", "model2-public-xgb-v1-shadow").strip()
MODEL2_HISTORY_DAYS = int(os.getenv("MODEL2_HISTORY_DAYS", "21"))
MODEL2_MIN_HOURLY_COVERAGE = float(
    os.getenv("MODEL2_MIN_HOURLY_COVERAGE", "0.50")
)

class InsufficientHistoryError(RuntimeError):
    pass

@lru_cache(maxsize=1)
def get_supabase():
    if not SUPABASE_URL:
        raise RuntimeError("SUPABASE_URL is not configured.")
    if not SUPABASE_SECRET_KEY:
        raise RuntimeError("SUPABASE_SECRET_KEY is not configured.")
    return create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)

booster = xgb.Booster()
booster.load_model(ROOT / "model2_xgboost.json")

def predict_probability(X: pd.DataFrame) -> float:
    X = X[FEATURE_COLUMNS].astype(np.float32)
    dm = xgb.DMatrix(X, feature_names=FEATURE_COLUMNS)
    return float(booster.predict(dm)[0])

def model_self_test_result():
    X = pd.DataFrame(
        [np.zeros(len(FEATURE_COLUMNS), dtype=np.float32)],
        columns=FEATURE_COLUMNS,
    )
    actual = predict_probability(X)
    expected = float(SELFTEST["expected_probability"])
    tolerance = float(SELFTEST["absolute_tolerance"])
    difference = abs(actual - expected)
    return {
        "passed": difference <= tolerance,
        "feature_count": len(FEATURE_COLUMNS),
        "expected_probability": expected,
        "actual_probability": actual,
        "absolute_difference": difference,
        "tolerance": tolerance,
    }

def require_internal_key(value):
    if not INTERNAL_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="INTERNAL_API_KEY is not configured.",
        )
    if value != INTERNAL_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid internal API key.",
        )

def fetch_recent_behavior(cow_id: str):
    start = (
        datetime.now(timezone.utc)
        - timedelta(days=MODEL2_HISTORY_DAYS)
    ).isoformat()

    columns = (
        "farm_id,cow_id,ts,"
        "walking_prop,grazing_prop,"
        "resting_prop,other_prop,"
        "activity_intensity,"
        "behavior_transition_rate,"
        "mean_model1_confidence,"
        "collar_temperature,"
        "data_quality"
    )

    all_rows = []

    page_size = 1000
    offset = 0

    while True:

        response = (
            get_supabase()
            .table("behavior_15min")
            .select(columns)
            .eq("cow_id", cow_id)
            .gte("ts", start)
            .order("ts")
            .range(
                offset,
                offset + page_size - 1
            )
            .execute()
        )

        batch = response.data or []

        all_rows.extend(batch)

        # Last page
        if len(batch) < page_size:
            break

        offset += page_size

    return all_rows

def prepare_hourly(rows):
    if not rows:
        raise InsufficientHistoryError(
            "No behavior_15min rows found for this cow."
        )

    df = pd.DataFrame(rows).copy()

    required = [
        "ts",
        "walking_prop",
        "grazing_prop",
        "resting_prop",
        "activity_intensity",
        "data_quality",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"behavior_15min rows are missing columns: {missing}"
        )

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    for c in [
        "walking_prop",
        "grazing_prop",
        "resting_prop",
        "activity_intensity",
        "data_quality",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(
        subset=[
            "ts",
            "walking_prop",
            "grazing_prop",
            "resting_prop",
            "activity_intensity",
        ]
    )

    if df.empty:
        raise InsufficientHistoryError(
            "No valid behavior_15min rows remain after cleaning."
        )

    df["local_ts"] = df["ts"].dt.tz_convert(FARM_TIMEZONE)
    df["hour_start"] = df["local_ts"].dt.floor("h")

    # Shadow-mode adapter:
    # public EAT/feeding signal <- Smart Herd grazing proxy.
    grouped = (
        df.groupby("hour_start", as_index=False)
        .agg(
            walking_prop=("walking_prop", "mean"),
            resting_prop=("resting_prop", "mean"),
            feeding_prop=("grazing_prop", "mean"),
            activity=("activity_intensity", "mean"),
            row_count=("ts", "size"),
            mean_15min_quality=("data_quality", "mean"),
        )
    )

    grouped["hourly_coverage"] = (
        grouped["row_count"].clip(upper=4) / 4.0
    ) * grouped["mean_15min_quality"].fillna(0.0)

    low_quality = grouped["hourly_coverage"] < MODEL2_MIN_HOURLY_COVERAGE
    grouped.loc[low_quality, PUBLIC_SIGNALS] = np.nan

    return grouped[
        ["hour_start", *PUBLIC_SIGNALS, "hourly_coverage"]
    ].copy()

def engineer_public_features(hourly):
    hourly = hourly.sort_values("hour_start").copy()

    start = hourly["hour_start"].min().floor("h")
    end = hourly["hour_start"].max().ceil("h")

    full_index = pd.date_range(
        start,
        end,
        freq="1h",
        tz=start.tz,
    )

    g = hourly.set_index("hour_start").reindex(full_index)
    g.index.name = "timestamp"

    g["observed"] = g["activity"].notna().astype(np.int8)

    for signal in PUBLIC_SIGNALS:
        g[signal] = (
            g[signal]
            .interpolate(
                method="linear",
                limit=2,
                limit_direction="both",
                limit_area="inside",
            )
            .astype(np.float32)
        )

    hour_of_day = g.index.hour
    g["hour_sin"] = np.sin(
        2.0 * np.pi * hour_of_day / 24.0
    ).astype(np.float32)
    g["hour_cos"] = np.cos(
        2.0 * np.pi * hour_of_day / 24.0
    ).astype(np.float32)

    count_columns = []

    for signal in PUBLIC_SIGNALS:
        baseline_med = pd.Series(np.nan, index=g.index, dtype=np.float32)
        baseline_scale = pd.Series(np.nan, index=g.index, dtype=np.float32)
        baseline_count = pd.Series(0.0, index=g.index, dtype=np.float32)

        fallback = max(
            float(np.nanstd(g[signal].values)) * 0.05,
            1e-4,
        )

        for slot in range(24):
            idx = g.index[hour_of_day == slot]
            x = g.loc[idx, signal]
            hist = x.shift(BASELINE_EXCLUSION_DAYS)

            roll = hist.rolling(
                BASELINE_DAYS,
                min_periods=MIN_BASELINE_DAYS,
            )

            med = roll.median()
            q25 = roll.quantile(0.25)
            q75 = roll.quantile(0.75)

            cnt = hist.rolling(
                BASELINE_DAYS,
                min_periods=1,
            ).count()

            scale = (q75 - q25) / 1.349
            scale = scale.where(scale >= 1e-5, fallback)

            baseline_med.loc[idx] = med.astype(np.float32)
            baseline_scale.loc[idx] = scale.astype(np.float32)
            baseline_count.loc[idx] = cnt.astype(np.float32)

        z = (
            (g[signal] - baseline_med)
            / baseline_scale.replace(0, np.nan)
        ).clip(-ROBUST_Z_CLIP, ROBUST_Z_CLIP)

        g[f"{signal}_z"] = z.astype(np.float32)

        count_name = f"{signal}_baseline_days"
        g[count_name] = baseline_count
        count_columns.append(count_name)

    g["baseline_days_available"] = (
        g[count_columns].min(axis=1).astype(np.float32)
    )

    g["baseline_quality"] = (
        g["baseline_days_available"] / BASELINE_DAYS
    ).clip(0, 1).astype(np.float32)

    for signal in PUBLIC_SIGNALS:
        zcol = f"{signal}_z"

        for horizon in HORIZONS_HOURS:
            min_periods = max(2, int(math.ceil(horizon * 0.5)))
            roll = g[zcol].rolling(
                horizon,
                min_periods=min_periods,
            )

            g[f"{zcol}_mean_{horizon}h"] = roll.mean().astype(np.float32)
            g[f"{zcol}_std_{horizon}h"] = roll.std().astype(np.float32)
            g[f"{zcol}_max_{horizon}h"] = roll.max().astype(np.float32)
            g[f"{zcol}_min_{horizon}h"] = roll.min().astype(np.float32)

        recent = g[zcol].rolling(6, min_periods=3).mean()
        g[f"{zcol}_trend_6h"] = (
            recent - recent.shift(6)
        ).astype(np.float32)

    g["activity_minus_resting_z"] = (
        g["activity_z"] - g["resting_prop_z"]
    ).astype(np.float32)

    g["walking_minus_resting_z"] = (
        g["walking_prop_z"] - g["resting_prop_z"]
    ).astype(np.float32)

    g["feeding_minus_resting_z"] = (
        g["feeding_prop_z"] - g["resting_prop_z"]
    ).astype(np.float32)

    for horizon in [6, 24, 48]:
        g[f"observation_fraction_{horizon}h"] = (
            g["observed"]
            .rolling(horizon, min_periods=1)
            .mean()
            .astype(np.float32)
        )

    return g

def build_latest_feature_row(rows):
    hourly = prepare_hourly(rows)
    engineered = engineer_public_features(hourly)

    observed_index = engineered.index[
        engineered["observed"] == 1
    ]

    if len(observed_index) == 0:
        raise InsufficientHistoryError(
            "No observed hourly record is available."
        )

    latest_ts = observed_index.max()
    row = engineered.loc[[latest_ts]].copy()

    baseline_days = float(
        row["baseline_days_available"].iloc[0]
    )

    if baseline_days < MIN_BASELINE_DAYS:
        raise InsufficientHistoryError(
            f"Only {baseline_days:.0f} baseline days are available; "
            f"at least {MIN_BASELINE_DAYS} are required."
        )

    missing_features = [
        c
        for c in FEATURE_COLUMNS
        if c not in row.columns
        or pd.isna(row[c].iloc[0])
    ]

    if missing_features:
        raise InsufficientHistoryError(
            "Latest row does not yet have a complete 85-feature vector. "
            f"Missing count: {len(missing_features)}."
        )

    X = row[FEATURE_COLUMNS].astype(np.float32)

    diagnostics = {
        "timestamp_local": latest_ts.isoformat(),
        "baseline_days_available": baseline_days,
        "baseline_quality": float(row["baseline_quality"].iloc[0]),
        "walking_z": float(row["walking_prop_z"].iloc[0]),
        "resting_z": float(row["resting_prop_z"].iloc[0]),
        "feeding_proxy_z": float(row["feeding_prop_z"].iloc[0]),
        "activity_z": float(row["activity_z"].iloc[0]),
    }

    return X, diagnostics

def persist_shadow_prediction(
    cow_id: str,
    probability: float,
    diagnostics: dict,
):
    payload = {
        "cow_id": cow_id,
        "ts": diagnostics["timestamp_local"],
        "probability": probability,
        "baseline_quality": diagnostics["baseline_quality"],
        "walking_z": diagnostics["walking_z"],
        "resting_z": diagnostics["resting_z"],
        "activity_z": diagnostics["activity_z"],
        "model_version": MODEL2_VERSION,
    }

    return (
        get_supabase()
        .table("estrus_predictions")
        .upsert(
            payload,
            on_conflict="cow_id,ts,model_version",
        )
        .execute()
        .data
    )

ADAPTER_WARNING = (
    "SHADOW MODE ONLY: the public XGBoost model was trained on hourly "
    "IN_ALLEYS/REST/EAT/ACTIVITY_LEVEL data. Live Smart Herd 15-minute "
    "walking/resting/grazing/activity summaries are aggregated to hourly; "
    "grazing is used only as a proxy for public feeding/EAT. "
    "Do not use this shadow probability as a production farmer alert."
)

app = FastAPI(
    title="Smart Herd AI Service",
    version="0.3.0",
    description=(
        "Flat Render deployment of Smart Herd Model 2 "
        "in public-benchmark shadow mode."
    ),
)

@app.get("/")
def root():
    return {
        "service": "smart-herd-ai",
        "status": "ok",
        "mode": "model2-public-shadow",
        "platform": "render",
    }

@app.get("/health")
def health():
    test = model_self_test_result()
    return {
        "status": "healthy" if test["passed"] else "model_self_test_failed",
        "model": "Smart Herd Model 2",
        "algorithm": PRODUCTION_CONFIG["selected_model"],
        "feature_count": len(FEATURE_COLUMNS),
        "supabase_configured": bool(
            SUPABASE_URL and SUPABASE_SECRET_KEY
        ),
        "internal_api_key_configured": bool(INTERNAL_API_KEY),
        "farm_timezone": FARM_TIMEZONE,
        "model_version": MODEL2_VERSION,
        "self_test": test,
    }

@app.post("/model2/self-test")
def model2_self_test(
    x_smart_herd_key: str | None = Header(default=None),
):
    require_internal_key(x_smart_herd_key)
    result = model_self_test_result()
    if not result["passed"]:
        raise HTTPException(status_code=500, detail=result)
    return result

@app.post("/model2/shadow/{cow_id}")
def model2_shadow(
    cow_id: str,
    persist: bool = Query(default=False),
    x_smart_herd_key: str | None = Header(default=None),
):
    require_internal_key(x_smart_herd_key)

    try:
        rows = fetch_recent_behavior(cow_id)
        X, diagnostics = build_latest_feature_row(rows)
        probability = predict_probability(X)

        output = {
            "status": "shadow_prediction",
            "cow_id": cow_id,
            "probability": probability,
            "threshold_reference": float(
                PRODUCTION_CONFIG["probability_threshold"]
            ),
            "model_version": MODEL2_VERSION,
            "feature_count": len(FEATURE_COLUMNS),
            "diagnostics": diagnostics,
            "adapter_warning": ADAPTER_WARNING,
            "persisted": False,
        }

        if persist:
            persist_shadow_prediction(
                cow_id,
                probability,
                diagnostics,
            )
            output["persisted"] = True

        return output

    except InsufficientHistoryError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "status": "insufficient_history",
                "cow_id": cow_id,
                "message": str(exc),
                "required_history_days": MODEL2_HISTORY_DAYS,
            },
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "cow_id": cow_id,
                "message": str(exc),
            },
        )
