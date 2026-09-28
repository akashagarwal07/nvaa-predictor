# backend/main.py  — FastAPI backend for NVAA Predictor
# Deploy to Render.com (free tier) — connects to React frontend on Vercel
# ──────────────────────────────────────────────────────────────
# Render setup:
#   Build command : pip install -r requirements.txt
#   Start command : uvicorn main:app --host 0.0.0.0 --port $PORT
# ──────────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import joblib, json, math, os, numpy as np

app = FastAPI(
    title="NVAA Cost Estimator API",
    description="Net Value of AI Adoption — ML-powered project cost predictor",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # restrict to your Vercel domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models at startup ────────────────────────────────────
MODEL_DIR = os.environ.get("MODEL_DIR", "./nvaa_models")
predictor = None
metadata  = {}

@app.on_event("startup")
def load_models():
    global predictor, metadata
    try:
        predictor = joblib.load(f"{MODEL_DIR}/nvaa_predictor.pkl")
        with open(f"{MODEL_DIR}/metadata.json") as f:
            metadata = json.load(f)
        print(f"✅  Models loaded from {MODEL_DIR}")
    except Exception as e:
        print(f"⚠️  Model load failed: {e}  — fallback to formula-only mode")


# ── Pydantic schemas ──────────────────────────────────────────
class ProjectParams(BaseModel):
    num_devs:             int   = Field(5,    ge=1,  le=100)
    total_sprints:        int   = Field(10,   ge=1,  le=100)
    sprint_length_days:   int   = Field(10,   ge=5,  le=30)
    domain_enc:           float = Field(1.0)
    methodology_enc:      float = Field(0.0)
    complexity:           float = Field(3.0,  ge=1,  le=5)
    genai_enabled:        int   = Field(1)
    genai_usage_intensity:float = Field(0.65, ge=0,  le=1)
    ai_maturity:          int   = Field(2,    ge=0,  le=3)
    # Auto-derived if omitted
    sprint_seq_norm:      Optional[float] = None
    log_num_devs:         Optional[float] = None
    ai_lift_factor:       Optional[float] = None
    quality_index:        Optional[float] = None
    sdlc_phase_num:       Optional[float] = None
    lag_productivity:     Optional[float] = None
    lag_tech_debt:        Optional[float] = None

class FinancialParams(BaseModel):
    labor_rate_usd_per_hour:      float = Field(75.0,   ge=10)
    defect_fix_cost_usd:          float = Field(2000.0, ge=0)
    ai_tool_cost_usd_per_dev_day: float = Field(0.50,   ge=0)
    setup_cost_usd:               float = Field(3000.0, ge=0)
    tech_debt_interest_rate:      float = Field(0.30,   ge=0, le=1)
    prompt_overhead_fraction:     float = Field(0.08,   ge=0, le=0.5)

class PredictRequest(BaseModel):
    project_params:  ProjectParams
    financial_params: FinancialParams


# ── Helper: auto-fill derived features ───────────────────────
def enrich(p: dict) -> dict:
    p = dict(p)
    if p.get("log_num_devs")  is None: p["log_num_devs"]  = math.log1p(p["num_devs"])
    if p.get("ai_lift_factor") is None:
        p["ai_lift_factor"] = (p["genai_enabled"]
                               * p["genai_usage_intensity"]
                               * p["ai_maturity"])
    if p.get("quality_index")  is None: p["quality_index"]  = 0.70
    if p.get("sdlc_phase_num") is None: p["sdlc_phase_num"] = 3.0
    if p.get("sprint_seq_norm")is None: p["sprint_seq_norm"]= 0.50
    pre = metadata.get("pre_baselines", {})
    if p.get("lag_productivity")is None:
        p["lag_productivity"] = pre.get("prod_base", 0.65)
    if p.get("lag_tech_debt")   is None:
        p["lag_tech_debt"]    = pre.get("tdr_base",  20.5)
    return p

def to_python(obj):
    """Recursively convert numpy types to native Python."""
    if isinstance(obj, dict):  return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [to_python(v) for v in obj]
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    return obj


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "NVAA Cost Estimator API", "status": "ok",
            "model_loaded": predictor is not None}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None,
            "metadata": {k: metadata.get(k) for k in
                         ["created_at","track_b_best","mean_apm","track_b_r2"]}}

@app.get("/metadata")
def get_metadata():
    return metadata

@app.post("/predict")
def predict(req: PredictRequest):
    """Full ML-based prediction (requires loaded model)."""
    if predictor is None:
        raise HTTPException(503, "Model not loaded — use /formula endpoint instead")
    try:
        params = enrich(req.project_params.model_dump())
        fp     = {k: v for k, v in req.financial_params.model_dump().items()
                  if k in {"labor_rate_usd_per_hour","defect_fix_cost_usd",
                            "ai_tool_cost_usd_per_dev_day","tech_debt_interest_rate",
                            "prompt_overhead_fraction"}}
        result = predictor.predict_nvaa(
            params, fp,
            n_sprints=req.project_params.total_sprints,
        )
        # Drop the sprint_records DataFrame (not JSON-serialisable)
        result.pop("sprint_records", None)
        return to_python(result)
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")

@app.post("/formula")
def formula_only(req: PredictRequest):
    """Formula-based fallback — no model required. Always available."""
    p  = req.project_params.model_dump()
    fp = req.financial_params.model_dump()

    # Simplified APM from domain + intensity + maturity
    domain_apm = metadata.get("domain_apm", {})
    base_apm   = metadata.get("mean_apm", 1.061)

    # derive apm
    apm = base_apm
    apm += (p["genai_usage_intensity"] - 0.567) * 0.10
    apm += (p["ai_maturity"]           - 1.7  ) * 0.015
    apm += (3.0 - p["complexity"]             ) * 0.008
    apm = max(0.98, min(2.0, apm))

    pre = metadata.get("pre_baselines", {"dd_base": 0.127, "tdr_base": 20.50})
    dd_pred  = max(0, pre.get("dd_base",  0.127) - p["genai_usage_intensity"] * 0.016)
    tdr_pred = max(0, pre.get("tdr_base", 20.50) - p["genai_usage_intensity"] * 9.17)

    sprints, cum = [], -fp["setup_cost_usd"]
    R   = fp["labor_rate_usd_per_hour"]
    sd  = p["sprint_length_days"]
    nd  = p["num_devs"]
    vel = nd * sd * 0.778 * apm

    for s in range(1, p["total_sprints"] + 1):
        eBase = nd * sd * 8
        leg   = eBase * (1 - 1/apm) * R
        qca   = max(0, pre.get("dd_base",0.127) - dd_pred) * vel * fp["defect_fix_cost_usd"] \
              + max(0, pre.get("tdr_base",20.5)  - tdr_pred) * 0.01 * eBase * R * fp.get("tech_debt_interest_rate", 0.30)
        tai   = nd * sd * fp["ai_tool_cost_usd_per_dev_day"] \
              + eBase * p["genai_usage_intensity"] * fp.get("prompt_overhead_fraction", 0.08) * R
        disc  = (leg + qca - tai) / (1.001 ** (s * sd / 7))
        cum  += disc
        sprints.append({"sprint": s, "LEG": round(leg,2), "QCA": round(qca,2),
                        "TAI": round(tai,2), "NVAA": round(leg+qca-tai,2), "cum": round(cum,2)})

    tot_leg = sum(s["LEG"] for s in sprints)
    tot_qca = sum(s["QCA"] for s in sprints)
    tot_tai = sum(s["TAI"] for s in sprints)
    roi     = cum / (tot_tai + fp["setup_cost_usd"]) * 100 if (tot_tai + fp["setup_cost_usd"]) else 0

    return {
        "apm": round(apm, 5), "tdr_predicted": round(tdr_pred, 3),
        "dd_predicted":  round(dd_pred, 5),
        "cumulative_nvaa": round(cum, 2),
        "roi_pct": round(roi, 2),
        "total_leg": round(tot_leg, 2),
        "total_qca": round(tot_qca, 2),
        "total_tai": round(tot_tai, 2),
        "sprints": sprints,
        "mode": "formula",
    }
