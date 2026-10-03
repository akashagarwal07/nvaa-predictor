# backend/main.py  —  NVAA Predictor API  (fixed deserialization)
# ──────────────────────────────────────────────────────────────
# Render setup:
#   Root directory  : backend
#   Build command   : pip install -r requirements.txt
#   Start command   : uvicorn main:app --host 0.0.0.0 --port $PORT
# ──────────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import joblib, json, os, math
import numpy as np

app = FastAPI(
    title="NVAA Cost Estimator API",
    description="Net Value of AI Adoption — ML-powered project cost predictor",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock down to your Vercel domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model directory ──────────────────────────────────────────
MODEL_DIR = os.environ.get("MODEL_DIR", "./nvaa_models")

# ── Module-level model state ─────────────────────────────────
prod_model    = None   # LGBMRegressor / RandomForest → predicts developer productivity
tdr_model     = None   # predicts technical debt ratio
dd_model      = None   # predicts defect density
mice_imputer  = None   # IterativeImputer (sklearn) — fills missing features
metadata      = {}

# ── Load on startup ──────────────────────────────────────────
@app.on_event("startup")
def load_models():
    global prod_model, tdr_model, dd_model, mice_imputer, metadata

    # Load metadata first (needed even if PKL files are missing)
    meta_path = f"{MODEL_DIR}/metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
        print(f"✅  Metadata loaded  |  features: {len(metadata.get('track_b_features',[]))}")
    else:
        print(f"⚠️  metadata.json not found at {meta_path}")
        return

    # Load individual model files — these are plain sklearn / LightGBM objects
    # and deserialize without any dependency on custom classes.
    try:
        prod_model   = joblib.load(f"{MODEL_DIR}/prod_model.pkl")
        print(f"✅  prod_model   loaded  ({type(prod_model).__name__})")
    except Exception as e:
        print(f"⚠️  prod_model   failed: {e}")

    try:
        tdr_model    = joblib.load(f"{MODEL_DIR}/tdr_model.pkl")
        print(f"✅  tdr_model    loaded  ({type(tdr_model).__name__})")
    except Exception as e:
        print(f"⚠️  tdr_model    failed: {e}")

    try:
        dd_model     = joblib.load(f"{MODEL_DIR}/dd_model.pkl")
        print(f"✅  dd_model     loaded  ({type(dd_model).__name__})")
    except Exception as e:
        print(f"⚠️  dd_model     failed: {e}")

    try:
        mice_imputer = joblib.load(f"{MODEL_DIR}/mice_imputer.pkl")
        print(f"✅  mice_imputer loaded  ({type(mice_imputer).__name__})")
    except Exception as e:
        print(f"⚠️  mice_imputer failed: {e}")


# ── Schemas ───────────────────────────────────────────────────
class ProjectParams(BaseModel):
    num_devs:               int   = Field(5,    ge=1,  le=100)
    total_sprints:          int   = Field(10,   ge=1,  le=100)
    sprint_length_days:     int   = Field(10,   ge=5,  le=30)
    domain_enc:             float = Field(0.0)   # FinTech=0,Healthcare=1,Manufacturing=2,Retail=3,Telecom=4
    methodology_enc:        float = Field(0.0)   # Agile=0,Hybrid=1,Waterfall=2
    complexity:             float = Field(3.0,  ge=1,  le=5)
    genai_enabled:          int   = Field(1)
    genai_usage_intensity:  float = Field(0.65, ge=0,  le=1)
    ai_maturity:            int   = Field(2,    ge=0,  le=3)
    # Optional — auto-derived if not provided
    sprint_seq_norm:        Optional[float] = None
    quality_index:          Optional[float] = None
    sdlc_phase_num:         Optional[float] = None
    lag_productivity:       Optional[float] = None
    lag_tech_debt:          Optional[float] = None

class FinancialParams(BaseModel):
    labor_rate_usd_per_hour:       float = Field(75.0)
    defect_fix_cost_usd:           float = Field(2000.0)
    ai_tool_cost_usd_per_dev_day:  float = Field(0.50)
    setup_cost_usd:                float = Field(3000.0)
    tech_debt_interest_rate:       float = Field(0.30)
    prompt_overhead_fraction:      float = Field(0.08)

class PredictRequest(BaseModel):
    project_params:   ProjectParams
    financial_params: FinancialParams


# ── Feature vector builder ────────────────────────────────────
def build_feature_vector(p: dict) -> np.ndarray:
    """
    Build the 16-column feature array in the exact order the models
    were trained on (TRACK_B_FEATURES from metadata.json).
    """
    pre = metadata.get("pre_baselines", {})
    feature_cols = metadata.get("track_b_features", [
        "num_devs","complexity","ai_maturity","total_sprints","sprint_length_days",
        "genai_enabled","genai_usage_intensity","sprint_seq_norm","log_num_devs",
        "ai_lift_factor","quality_index","lag_productivity","lag_tech_debt",
        "domain_enc","methodology_enc","sdlc_phase_num",
    ])

    derived = {
        "log_num_devs":    math.log1p(p["num_devs"]),
        "ai_lift_factor":  float(p["genai_enabled"])
                           * p["genai_usage_intensity"]
                           * float(p["ai_maturity"]),
        "sprint_seq_norm": p.get("sprint_seq_norm") or 0.50,
        "quality_index":   p.get("quality_index")   or 0.70,
        "sdlc_phase_num":  p.get("sdlc_phase_num")  or 3.0,
        "lag_productivity":p.get("lag_productivity") or pre.get("prod_base", 0.65),
        "lag_tech_debt":   p.get("lag_tech_debt")    or pre.get("tdr_base",  20.5),
    }
    merged = {**p, **{k: v for k, v in derived.items() if p.get(k) is None}}

    arr = np.array([merged.get(col, 0.0) for col in feature_cols], dtype=float)
    return arr.reshape(1, -1)


def impute(X: np.ndarray) -> np.ndarray:
    """Apply MICE imputer if loaded; return X unchanged otherwise."""
    if mice_imputer is not None:
        try:
            return mice_imputer.transform(X)
        except Exception:
            pass
    return X


# ── Core prediction logic ─────────────────────────────────────
def ml_predict(p: dict) -> dict:
    """
    Predict APM, TDR, DD using the loaded ML models.
    APM is computed via counterfactual: predict WITH AI / predict WITHOUT AI.
    """
    if prod_model is None:
        raise HTTPException(503, "prod_model not loaded")

    # Feature vector WITH AI
    X_ai    = impute(build_feature_vector(p))

    # Counterfactual — zero out all AI-related features
    p_no_ai = dict(p)
    p_no_ai.update(genai_enabled=0, genai_usage_intensity=0.0, ai_maturity=0)
    X_no_ai = impute(build_feature_vector(p_no_ai))

    prod_ai    = float(prod_model.predict(X_ai)[0])
    prod_no_ai = float(prod_model.predict(X_no_ai)[0])
    apm        = float(np.clip(prod_ai / max(prod_no_ai, 1e-6), 0.5, 5.0))

    tdr = float(tdr_model.predict(X_ai)[0]) if tdr_model is not None else None
    dd  = float(dd_model.predict(X_ai)[0])  if dd_model  is not None else None

    return {
        "apm":             round(apm,    5),
        "tdr_predicted":   round(max(tdr, 0), 3) if tdr is not None else None,
        "dd_predicted":    round(max(dd,  0), 5) if dd  is not None else None,
        "productivity_ai":   round(prod_ai,    4),
        "productivity_base": round(prod_no_ai, 4),
        "model_r2":        metadata.get("track_b_r2", {}).get("productivity"),
    }


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message":      "NVAA Cost Estimator API",
        "status":       "ok",
        "model_loaded": prod_model is not None,
    }


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": prod_model is not None,
        "models": {
            "prod_model":   prod_model  is not None,
            "tdr_model":    tdr_model   is not None,
            "dd_model":     dd_model    is not None,
            "mice_imputer": mice_imputer is not None,
        },
        "metadata": {k: metadata.get(k) for k in
                     ["created_at","track_b_best","mean_apm","track_b_r2"]},
    }


@app.get("/metadata")
def get_metadata():
    return metadata


@app.post("/predict")
def predict(req: PredictRequest):
    """
    ML-based prediction.
    Returns APM (and quality deltas) from the trained Random Forest model.
    The frontend computes the sprint breakdown from these values.
    """
    if prod_model is None:
        raise HTTPException(
            503,
            detail={
                "error": "ML models not loaded",
                "hint":  "Run Cell 12 in Colab, download nvaa_models/, push to GitHub, redeploy Render.",
            }
        )
    try:
        p = req.project_params.model_dump()
        return ml_predict(p)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Prediction error: {str(e)}")


@app.post("/formula")
def formula_fallback(req: PredictRequest):
    """
    Lightweight formula-based fallback — always available even without models.
    The JS frontend uses this automatically when /predict returns 503.
    """
    p  = req.project_params.model_dump()
    fp = req.financial_params.model_dump()
    pre = metadata.get("pre_baselines", {"dd_base": 0.127, "tdr_base": 20.50})

    # Simplified APM
    base = metadata.get("domain_apm", {}).get(str(p["domain_enc"]),
                 metadata.get("mean_apm", 1.061))
    apm  = float(np.clip(
        base
        + (p["genai_usage_intensity"] - 0.567) * 0.10
        + (p["ai_maturity"]           - 1.700) * 0.015
        + (3.0 - p["complexity"])               * 0.008
        - max(0, p["num_devs"] - 5)             * 0.004,
        0.98, 2.0,
    ))
    tdr = max(0.0, pre.get("tdr_base", 20.50) - p["genai_usage_intensity"] * 9.171)
    dd  = max(0.0, pre.get("dd_base",  0.127) - p["genai_usage_intensity"] * 0.0159)

    return {
        "apm":           round(apm, 5),
        "tdr_predicted": round(tdr, 3),
        "dd_predicted":  round(dd,  5),
        "mode":          "formula",
    }
