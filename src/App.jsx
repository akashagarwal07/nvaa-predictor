import { useState } from "react"
import {
  BarChart, Bar, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine,
} from "recharts"

// ═══════════════════════════════════════════════════════════════
//  ⚙️  CONFIG
//  After deploying backend to Render, replace the URL below.
//  Leave as-is → app runs entirely on the built-in JS formula.
// ═══════════════════════════════════════════════════════════════
const API_URL = "https://your-backend.onrender.com"

// ═══════════════════════════════════════════════════════════════
//  CONSTANTS  (calibrated from 198 real sprint observations)
// ═══════════════════════════════════════════════════════════════
const DOMAIN_APM = {
  FinTech:1.072, Healthcare:1.046, Manufacturing:1.094,
  Telecom:1.009, Retail:1.050,     Other:1.061,
}
const DOMAIN_ENC = {
  FinTech:0, Healthcare:1, Manufacturing:2,
  Retail:3,  Telecom:4,    Other:-1,
}
const DOMAIN_COLOR = {
  FinTech:"#6366F1", Healthcare:"#0EA5E9", Manufacturing:"#F59E0B",
  Telecom:"#10B981", Retail:"#EC4899",     Other:"#6B7280",
}
const DOMAINS  = Object.keys(DOMAIN_APM)
const PRE_DD   = 0.127
const PRE_TDR  = 20.50

// ═══════════════════════════════════════════════════════════════
//  FORMULA ENGINE  (JS fallback when API is unavailable)
// ═══════════════════════════════════════════════════════════════
function formulaAPM({ intensity, maturity, complexity, numDevs, domain }) {
  const base = DOMAIN_APM[domain] ?? 1.061
  return Math.max(0.98, Math.min(2.0,
    base
    + (intensity - 0.567) * 0.10
    + (maturity  - 1.700) * 0.015
    + (3.0 - complexity)  * 0.008
    - Math.max(0, numDevs - 5) * 0.004
  ))
}

function buildSprints({ numDevs, sprintDays, totalSprints, intensity,
                         laborRate, defectCost, aiCost, setupCost,
                         apm, tdrPred, ddPred }) {
  const vel = numDevs * sprintDays * 0.778 * apm
  const rows = []
  let cum = -setupCost

  for (let s = 1; s <= totalSprints; s++) {
    const E   = numDevs * sprintDays * 8
    const leg = E * (1 - 1 / apm) * laborRate
    const qca = Math.max(0, PRE_DD  - ddPred)  * vel * defectCost
              + Math.max(0, PRE_TDR - tdrPred) * 0.01 * E * laborRate * 0.30
    const tai = numDevs * sprintDays * aiCost
              + E * intensity * 0.08 * laborRate
    cum += (leg + qca - tai) / Math.pow(1.001, s * sprintDays / 7)
    rows.push({
      s,
      leg: +leg.toFixed(0), qca: +qca.toFixed(0),
      tai: +tai.toFixed(0), net: +(leg+qca-tai).toFixed(0),
      cum: +cum.toFixed(0),
    })
  }

  const tL = rows.reduce((a,r) => a + r.leg, 0)
  const tQ = rows.reduce((a,r) => a + r.qca, 0)
  const tT = rows.reduce((a,r) => a + r.tai, 0)
  const be = rows.findIndex(r => r.cum >= 0)

  return {
    rows, cum, tL, tQ, tT,
    roi:       cum / (tT + setupCost) * 100,
    effortH:   rows.reduce((a,r) => a + r.leg / laborRate, 0),
    breakEven: be >= 0 ? be + 1 : null,
  }
}

// ═══════════════════════════════════════════════════════════════
//  API CALL  →  POST /predict to FastAPI on Render
// ═══════════════════════════════════════════════════════════════
async function callAPI(p1, p2, p3) {
  // Skip if URL has not been configured yet
  if (!API_URL || API_URL.includes("your-backend")) return null

  try {
    const resp = await fetch(`${API_URL}/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_params: {
          num_devs:               p1.numDevs,
          total_sprints:          p1.totalSprints,
          sprint_length_days:     p1.sprintDays,
          domain_enc:             DOMAIN_ENC[p1.domain] ?? 0,
          methodology_enc:        0,           // Agile default
          complexity:             p1.complexity,
          genai_enabled:          p2.enabled ? 1 : 0,
          genai_usage_intensity:  p2.intensity,
          ai_maturity:            p2.maturity,
          // optional — backend auto-derives these if omitted:
          sprint_seq_norm:        null,
          log_num_devs:           null,
          ai_lift_factor:         null,
          quality_index:          null,
          sdlc_phase_num:         null,
          lag_productivity:       null,
          lag_tech_debt:          null,
        },
        financial_params: {
          labor_rate_usd_per_hour:       p3.laborRate,
          defect_fix_cost_usd:           p3.defectCost,
          ai_tool_cost_usd_per_dev_day:  p3.aiCost,
          setup_cost_usd:                p3.setupCost,
          tech_debt_interest_rate:       0.30,
          prompt_overhead_fraction:      0.08,
        },
      }),
    })

    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${resp.statusText}`)
    return await resp.json()
  } catch (err) {
    console.warn("[NVAA] API unavailable → falling back to JS formula.", err.message)
    return null
  }
}

// ═══════════════════════════════════════════════════════════════
//  ORCHESTRATE  →  merge API result (ML model) with local sprints
// ═══════════════════════════════════════════════════════════════
async function computeResult(p1, p2, p3) {
  const api = await callAPI(p1, p2, p3)

  // Use ML model predictions if API responded, otherwise formula
  const apm     = api?.apm            ?? formulaAPM({ ...p2, ...p1 })
  const tdrPred = api?.tdr_predicted  ?? Math.max(0, PRE_TDR - p2.intensity * 9.171)
  const ddPred  = api?.dd_predicted   ?? Math.max(0, PRE_DD  - p2.intensity * 0.0159)

  // Sprint-by-sprint breakdown always computed client-side (for charts)
  const sprints = buildSprints({ ...p1, ...p2, ...p3, apm, tdrPred, ddPred })

  return {
    ...sprints,
    apm, tdrPred, ddPred,
    source: api ? "ml" : "formula",   // shown as badge in Results
    modelR2: api ? 0.728 : null,
  }
}

// ═══════════════════════════════════════════════════════════════
//  FORMATTERS
// ═══════════════════════════════════════════════════════════════
const $   = n => new Intl.NumberFormat("en-US",{style:"currency",currency:"USD",maximumFractionDigits:0}).format(n)
const fix = (n, d = 3) => Number(n).toFixed(d)
const pct = (n, d = 1) => `${n >= 0 ? "+" : ""}${Number(n).toFixed(d)}%`

// ═══════════════════════════════════════════════════════════════
//  DEFAULT FORM STATE
// ═══════════════════════════════════════════════════════════════
const D1 = { numDevs:5, totalSprints:10, sprintDays:10, domain:"FinTech", complexity:3 }
const D2 = { enabled:true, intensity:0.65, maturity:2 }
const D3 = { laborRate:75, defectCost:2000, aiCost:0.50, setupCost:3000 }

// ═══════════════════════════════════════════════════════════════
//  DESIGN TOKENS
// ═══════════════════════════════════════════════════════════════
const C = {
  leg: "#10B981",   // emerald
  qca: "#F59E0B",   // amber
  tai: "#EF4444",   // red
  cum: "#6366F1",   // indigo
}

// ═══════════════════════════════════════════════════════════════
//  UI ATOMS
// ═══════════════════════════════════════════════════════════════
function RangeSlider({ label, value, min, max, step = 1, onChange, display }) {
  const pct = ((value - min) / (max - min)) * 100
  return (
    <div className="mb-5">
      <div className="flex justify-between items-baseline mb-2">
        <span className="text-sm font-semibold text-gray-700">{label}</span>
        <span className="text-sm font-bold text-indigo-600 bg-indigo-50 rounded-lg px-2.5 py-0.5">
          {display(value)}
        </span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(+e.target.value)}
        className="w-full h-2 rounded-full appearance-none cursor-pointer outline-none"
        style={{ background: `linear-gradient(to right,#6366F1 ${pct}%,#E5E7EB ${pct}%)` }}
      />
      <div className="flex justify-between text-xs text-gray-400 mt-1">
        <span>{display(min)}</span><span>{display(max)}</span>
      </div>
    </div>
  )
}

function MoneyInput({ label, hint, value, min = 0, max, step = 1, onChange, prefix = "$", suffix = "" }) {
  return (
    <div className="mb-4">
      <div className="flex justify-between items-baseline mb-1.5">
        <span className="text-sm font-semibold text-gray-700">{label}</span>
        {hint && <span className="text-xs text-gray-400 italic">{hint}</span>}
      </div>
      <div className="flex border-2 border-gray-200 rounded-xl overflow-hidden focus-within:border-indigo-400 transition-colors bg-white">
        {prefix && (
          <span className="px-3 py-2.5 bg-gray-50 text-gray-500 font-bold text-sm border-r-2 border-gray-200 flex items-center select-none">
            {prefix}
          </span>
        )}
        <input
          type="number" value={value} min={min} max={max} step={step}
          onChange={e => onChange(+e.target.value)}
          className="flex-1 px-3 py-2.5 text-right font-bold text-gray-900 text-sm outline-none bg-white min-w-0"
        />
        {suffix && (
          <span className="px-3 py-2.5 bg-gray-50 text-gray-500 font-medium text-sm border-l-2 border-gray-200 flex items-center select-none">
            {suffix}
          </span>
        )}
      </div>
    </div>
  )
}

function Block({ title, icon, children }) {
  return (
    <div className="bg-white rounded-2xl border-2 border-gray-100 p-5 mb-4 shadow-sm">
      {title && (
        <p className="text-xs font-black uppercase tracking-widest text-indigo-500 mb-4 flex items-center gap-2">
          <span className="text-base">{icon}</span>{title}
        </p>
      )}
      {children}
    </div>
  )
}

function Nav({ onBack, onNext, nextLabel = "Next →", loading = false, disabled = false }) {
  return (
    <div className="flex gap-3 mt-6">
      {onBack && (
        <button onClick={onBack}
          className="flex-1 py-3.5 rounded-xl border-2 border-gray-200 text-gray-600 font-semibold hover:border-indigo-300 hover:text-indigo-600 transition-all">
          ← Back
        </button>
      )}
      <button onClick={onNext} disabled={loading || disabled}
        className="flex-1 py-3.5 rounded-xl bg-indigo-600 text-white font-bold hover:bg-indigo-700 disabled:opacity-50 transition-all flex items-center justify-center gap-2">
        {loading && (
          <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 100 16v-4l-3 3 3 3v-4a8 8 0 01-8-8z"/>
          </svg>
        )}
        {loading ? "Calculating…" : nextLabel}
      </button>
    </div>
  )
}

function Steps({ current }) {
  const steps = [
    { n:1, label:"Project" },
    { n:2, label:"AI Setup" },
    { n:3, label:"Financials" },
    { n:4, label:"Results" },
  ]
  return (
    <div className="flex items-center mb-8">
      {steps.map(({ n, label }, i) => (
        <div key={n} className="flex items-center flex-1 last:flex-none">
          <div className="flex flex-col items-center gap-1">
            <div className={`w-9 h-9 rounded-full flex items-center justify-center text-sm font-black transition-all
              ${n < current  ? "bg-indigo-600 text-white"
              : n === current ? "bg-indigo-600 text-white ring-4 ring-indigo-100"
              :                 "bg-gray-100 text-gray-400"}`}>
              {n < current ? "✓" : n}
            </div>
            <span className={`text-xs font-bold hidden sm:block transition-colors
              ${n <= current ? "text-indigo-600" : "text-gray-400"}`}>
              {label}
            </span>
          </div>
          {i < steps.length - 1 && (
            <div className={`flex-1 h-0.5 mx-2 rounded-full transition-all ${n < current ? "bg-indigo-500" : "bg-gray-200"}`} />
          )}
        </div>
      ))}
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════
//  STEP 1 — Project
// ═══════════════════════════════════════════════════════════════
function Step1({ d, set, onNext }) {
  return (
    <div>
      <div className="mb-6">
        <p className="text-xs font-black text-indigo-500 uppercase tracking-widest mb-1">Step 1 of 3</p>
        <h2 className="text-2xl font-black text-gray-900">Project Setup</h2>
        <p className="text-gray-500 text-sm mt-1">Tell us about your team and project scope.</p>
      </div>

      <Block title="Team & Timeline" icon="👥">
        <RangeSlider label="Number of Developers" value={d.numDevs} min={1} max={25}
          onChange={v => set({...d, numDevs:v})} display={v=>`${v} devs`} />
        <RangeSlider label="Total Sprints" value={d.totalSprints} min={2} max={30}
          onChange={v => set({...d, totalSprints:v})} display={v=>`${v} sprints`} />
        <RangeSlider label="Sprint Length" value={d.sprintDays} min={5} max={30} step={5}
          onChange={v => set({...d, sprintDays:v})} display={v=>`${v} days`} />
      </Block>

      <Block title="Project Context" icon="🏢">
        <div className="mb-5">
          <p className="text-sm font-semibold text-gray-700 mb-3">Industry Domain</p>
          <div className="grid grid-cols-3 gap-2">
            {DOMAINS.map(dm => {
              const active = d.domain === dm
              const color  = DOMAIN_COLOR[dm]
              return (
                <button key={dm} onClick={() => set({...d, domain:dm})}
                  className={`py-3 px-2 rounded-xl text-sm font-bold border-2 transition-all
                    ${active ? "text-white border-transparent" : "bg-white text-gray-600 border-gray-200 hover:border-gray-300"}`}
                  style={active ? { background: color, borderColor: color } : {}}>
                  {dm}
                </button>
              )
            })}
          </div>
        </div>
        <RangeSlider label="Project Complexity" value={d.complexity} min={1} max={5} step={1}
          onChange={v => set({...d, complexity:v})}
          display={v=>["","Simple","Low","Medium","High","Critical"][v]||v} />
      </Block>

      <Nav onNext={onNext} nextLabel="Configure AI →" />
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════
//  STEP 2 — AI Configuration
// ═══════════════════════════════════════════════════════════════
function Step2({ d, proj, set, onNext, onBack }) {
  const apmPreview = formulaAPM({
    intensity: d.intensity, maturity: d.maturity,
    complexity: proj.complexity, numDevs: proj.numDevs, domain: proj.domain,
  })

  return (
    <div>
      <div className="mb-6">
        <p className="text-xs font-black text-indigo-500 uppercase tracking-widest mb-1">Step 2 of 3</p>
        <h2 className="text-2xl font-black text-gray-900">AI Configuration</h2>
        <p className="text-gray-500 text-sm mt-1">How is your team adopting GenAI tools?</p>
      </div>

      <Block title="Adoption Status" icon="🤖">
        <p className="text-sm font-semibold text-gray-700 mb-2">GenAI Tools Enabled?</p>
        <div className="flex gap-2 mb-5">
          {[
            [true,  "✅  Yes, using GenAI",  "#10B981"],
            [false, "❌  No, not using",     "#EF4444"],
          ].map(([val, label, color]) => (
            <button key={String(val)} onClick={() => set({...d, enabled:val})}
              className={`flex-1 py-3 rounded-xl text-sm font-bold border-2 transition-all
                ${d.enabled === val ? "text-white border-transparent" : "bg-white text-gray-600 border-gray-200"}`}
              style={d.enabled === val ? { background: color } : {}}>
              {label}
            </button>
          ))}
        </div>

        {d.enabled && (
          <>
            <RangeSlider label="GenAI Usage Intensity"
              value={d.intensity} min={0} max={1} step={0.05}
              onChange={v => set({...d, intensity:+v.toFixed(2)})}
              display={v=>`${(v*100).toFixed(0)}% of tasks`} />

            <div className="mb-5">
              <p className="text-sm font-semibold text-gray-700 mb-3">Team AI Maturity</p>
              <div className="grid grid-cols-3 gap-2">
                {[
                  [1, "Level 1", "Beginner", "🌱"],
                  [2, "Level 2", "Intermediate", "🌿"],
                  [3, "Level 3", "Expert", "🌳"],
                ].map(([m, lbl, desc, icon]) => (
                  <button key={m} onClick={() => set({...d, maturity:m})}
                    className={`py-3 rounded-xl border-2 transition-all text-center
                      ${d.maturity === m
                        ? "bg-indigo-600 text-white border-indigo-600"
                        : "bg-white text-gray-600 border-gray-200 hover:border-indigo-300"}`}>
                    <p className="text-lg">{icon}</p>
                    <p className="text-sm font-bold">{lbl}</p>
                    <p className={`text-xs ${d.maturity === m ? "text-indigo-200" : "text-gray-400"}`}>{desc}</p>
                  </button>
                ))}
              </div>
            </div>
          </>
        )}
      </Block>

      {d.enabled && (
        <div className="rounded-2xl overflow-hidden mb-4 shadow-sm border border-indigo-100">
          <div className="bg-gradient-to-r from-indigo-600 to-violet-600 px-5 py-3">
            <p className="text-xs font-bold text-indigo-200 uppercase tracking-widest">Live APM Preview</p>
          </div>
          <div className="bg-indigo-50 px-5 py-4 flex items-end justify-between">
            <div>
              <p className="text-5xl font-black text-indigo-800">{fix(apmPreview)}×</p>
              <p className="text-sm text-indigo-500 font-semibold mt-1">
                {pct((apmPreview - 1)*100, 1)} productivity uplift vs no-AI
              </p>
            </div>
            <div className="text-right text-xs text-indigo-400 max-w-36">
              <p className="font-bold">SHAP-calibrated formula</p>
              <p>Full ML model (R²=0.728) applied at runtime when backend is connected</p>
            </div>
          </div>
        </div>
      )}

      <Nav onBack={onBack} onNext={onNext}
        nextLabel="Set Financials →"
        disabled={d.enabled && d.intensity === 0} />
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════
//  STEP 3 — Financial Parameters
// ═══════════════════════════════════════════════════════════════
function Step3({ d, set, onNext, onBack, loading }) {
  return (
    <div>
      <div className="mb-6">
        <p className="text-xs font-black text-indigo-500 uppercase tracking-widest mb-1">Step 3 of 3</p>
        <h2 className="text-2xl font-black text-gray-900">Financial Parameters</h2>
        <p className="text-gray-500 text-sm mt-1">Enter your organisation's actual cost figures.</p>
      </div>

      <Block title="Labour & Defect Costs" icon="💰">
        <MoneyInput label="Fully-Loaded Labour Rate" hint="salary + benefits + overhead"
          value={d.laborRate} min={10} max={300} step={5}
          onChange={v => set({...d, laborRate:v})} prefix="$/hr" />
        <MoneyInput label="Avg. Production Defect Fix Cost" hint="per escaped defect"
          value={d.defectCost} min={100} max={50000} step={100}
          onChange={v => set({...d, defectCost:v})} prefix="$" suffix="/ defect" />
        <div className="text-xs text-gray-400 bg-gray-50 rounded-xl p-3 -mt-1">
          💡 Industry range: $500 (minor bug) – $20,000 (critical production outage) · Jones (2012)
        </div>
      </Block>

      <Block title="AI Tooling Costs" icon="⚙️">
        <MoneyInput label="AI Tool Cost Per Developer" hint="per working day"
          value={d.aiCost} min={0.05} max={10} step={0.05}
          onChange={v => set({...d, aiCost:+v.toFixed(2)})}
          prefix="$" suffix="/ dev / day" />
        <div className="text-xs text-gray-400 bg-gray-50 rounded-xl p-3 mb-4 -mt-1">
          GitHub Copilot ≈ $0.45 · ChatGPT Plus ≈ $0.65 · Enterprise tier ≈ $1.30+
        </div>
        <MoneyInput label="One-Time Setup Cost" hint="licences + integration + training"
          value={d.setupCost} min={0} max={50000} step={500}
          onChange={v => set({...d, setupCost:v})} prefix="$" />
      </Block>

      <Nav onBack={onBack} onNext={onNext} loading={loading}
        nextLabel="Calculate NVAA →" />
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════
//  RESULTS DASHBOARD
// ═══════════════════════════════════════════════════════════════
function KPICard({ label, value, sub, bg, text, border }) {
  return (
    <div className={`rounded-2xl p-4 border-2 ${bg} ${border}`}>
      <p className={`text-xs font-bold uppercase tracking-wide opacity-60 ${text}`}>{label}</p>
      <p className={`text-2xl font-black mt-1 ${text}`}>{value}</p>
      {sub && <p className={`text-xs mt-1 opacity-70 ${text}`}>{sub}</p>}
    </div>
  )
}

function ChartTip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-lg p-3 text-xs">
      <p className="font-bold text-gray-700 mb-1.5">Sprint {label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }} className="font-semibold">{p.name}: {$(p.value)}</p>
      ))}
    </div>
  )
}

function Results({ result, proj, ai, fin, onReset }) {
  if (!result) return null
  const { rows, cum, tL, tQ, tT, roi, effortH, breakEven, apm, tdrPred, source, modelR2 } = result
  const pos = cum >= 0

  const barData = [
    { name:"LEG",      value:+tL.toFixed(0) },
    { name:"QCA",      value:+tQ.toFixed(0) },
    { name:"TAI",      value:+tT.toFixed(0) },
    { name:"Net NVAA", value:+cum.toFixed(0) },
  ]
  const barColors = [C.leg, C.qca, C.tai, pos ? C.cum : "#EF4444"]

  const pbData = rows.map(r => ({ sprint:`S${r.s}`, LEG:r.leg, QCA:r.qca, TAI:-r.tai }))
  const cumData = rows.map(r => ({ sprint:`S${r.s}`, NVAA:r.cum }))

  return (
    <div>
      {/* Header row */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="flex items-center gap-2 mb-0.5">
            <h2 className="text-xl font-black text-gray-900">Results</h2>
            <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${
              source === "ml"
                ? "bg-indigo-100 text-indigo-700"
                : "bg-amber-100 text-amber-700"}`}>
              {source === "ml" ? `🧠 ML Model (R²=${modelR2})` : "📐 JS Formula"}
            </span>
          </div>
          <p className="text-xs text-gray-400">
            {proj.numDevs} devs · {proj.totalSprints} sprints · {proj.domain} · {ai.intensity*100|0}% intensity
          </p>
        </div>
        <button onClick={onReset}
          className="text-xs font-bold text-indigo-600 hover:underline whitespace-nowrap">
          ← Edit
        </button>
      </div>

      {/* Hero */}
      <div className={`rounded-2xl p-6 mb-4 text-center shadow-lg text-white
        ${pos ? "bg-gradient-to-br from-emerald-500 to-green-700"
               : "bg-gradient-to-br from-red-500 to-rose-700"}`}>
        <p className="text-xs font-bold uppercase tracking-widest opacity-80 mb-1">
          Cumulative NVAA — {proj.totalSprints} Sprints
        </p>
        <p className="text-6xl font-black">{$(cum)}</p>
        <div className="flex items-center justify-center gap-4 mt-3 text-sm opacity-90 flex-wrap">
          <span>ROI&nbsp;<strong>{fix(roi,1)}%</strong></span>
          <span className="opacity-40">|</span>
          <span>APM&nbsp;<strong>{fix(apm)}×</strong></span>
          <span className="opacity-40">|</span>
          <span>{breakEven ? `Breakeven Sprint ${breakEven}` : "No breakeven in project"}</span>
        </div>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <KPICard label="AI Productivity Multiplier"
          value={`${fix(apm)}×`}
          sub={`${pct((apm-1)*100,1)} vs no-AI`}
          bg="bg-indigo-50" text="text-indigo-900" border="border-indigo-200" />
        <KPICard label="Return on Investment"
          value={`${fix(roi,1)}%`}
          sub="net NVAA ÷ total AI spend"
          bg={roi>=0?"bg-emerald-50":"bg-red-50"}
          text={roi>=0?"text-emerald-900":"text-red-900"}
          border={roi>=0?"border-emerald-200":"border-red-200"} />
        <KPICard label="Labour Savings (LEG)"
          value={$(tL)}
          sub={`${fix(effortH,0)} person-hours recovered`}
          bg="bg-emerald-50" text="text-emerald-900" border="border-emerald-200" />
        <KPICard label="Quality Savings (QCA)"
          value={$(tQ)}
          sub={`Tech debt: ${PRE_TDR}% → ${fix(tdrPred,1)}%`}
          bg="bg-amber-50" text="text-amber-900" border="border-amber-200" />
      </div>

      {/* Total breakdown */}
      <Block>
        <p className="text-xs font-black uppercase tracking-widest text-gray-400 mb-4">
          💰 Project-Level Breakdown
        </p>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={barData} layout="vertical" margin={{ left:8, right:60 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#F3F4F6" />
            <XAxis type="number" tickFormatter={v=>`$${(v/1000).toFixed(0)}k`}
              tick={{ fontSize:10, fill:"#9CA3AF" }} />
            <YAxis type="category" dataKey="name" width={72}
              tick={{ fontSize:12, fontWeight:700, fill:"#374151" }} />
            <Tooltip formatter={v=>[$(v)]} />
            <Bar dataKey="value" radius={[0,8,8,0]}
              label={{ position:"right", formatter:v=>$(v), fontSize:11, fontWeight:700, fill:"#374151" }}>
              {barData.map((_, i) => <Cell key={i} fill={barColors[i]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="flex justify-center gap-5 mt-2 text-xs flex-wrap">
          {[["LEG",C.leg,"Labour Savings"],["QCA",C.qca,"Quality Savings"],["TAI",C.tai,"AI Cost"]].map(([k,c,l])=>(
            <div key={k} className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded" style={{ background:c }}/>
              <span className="text-gray-500 font-medium">{l}</span>
            </div>
          ))}
        </div>
      </Block>

      {/* Per-sprint stacked */}
      <Block>
        <p className="text-xs font-black uppercase tracking-widest text-gray-400 mb-4">
          📊 Per-Sprint Components
        </p>
        <ResponsiveContainer width="100%" height={170}>
          <BarChart data={pbData} margin={{ right:8 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F3F4F6" />
            <XAxis dataKey="sprint" tick={{ fontSize:10, fill:"#9CA3AF" }} />
            <YAxis tickFormatter={v=>`$${(v/1000).toFixed(0)}k`} tick={{ fontSize:10, fill:"#9CA3AF" }} />
            <Tooltip content={<ChartTip />} />
            <Bar dataKey="LEG" stackId="a" fill={C.leg} name="LEG (Labour)" />
            <Bar dataKey="QCA" stackId="a" fill={C.qca} name="QCA (Quality)" />
            <Bar dataKey="TAI" stackId="a" fill={C.tai} name="TAI (AI Cost)" />
          </BarChart>
        </ResponsiveContainer>
      </Block>

      {/* Cumulative line */}
      <Block>
        <p className="text-xs font-black uppercase tracking-widest text-gray-400 mb-4">
          📈 Cumulative NVAA
        </p>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={cumData} margin={{ right:8 }}>
            <defs>
              <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor={pos ? C.cum : "#EF4444"} stopOpacity={0.3}/>
                <stop offset="95%" stopColor={pos ? C.cum : "#EF4444"} stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" />
            <XAxis dataKey="sprint" tick={{ fontSize:10, fill:"#9CA3AF" }} />
            <YAxis tickFormatter={v=>`$${(v/1000).toFixed(0)}k`} tick={{ fontSize:10, fill:"#9CA3AF" }} />
            <Tooltip formatter={v=>[$(v),"Cumulative NVAA"]} labelFormatter={l=>`Sprint ${l.replace("S","")}`} />
            <ReferenceLine y={0} stroke="#9CA3AF" strokeDasharray="4 4" />
            <Area type="monotone" dataKey="NVAA"
              stroke={pos ? C.cum : "#EF4444"} strokeWidth={2.5}
              fill="url(#areaGrad)" dot={{ r:3, fill: pos ? C.cum : "#EF4444" }} activeDot={{ r:5 }} />
          </AreaChart>
        </ResponsiveContainer>
        {breakEven && (
          <p className="text-xs text-center text-gray-400 mt-2">
            Setup cost recovered at Sprint {breakEven} · Terminal value {$(cum)}
          </p>
        )}
      </Block>

      {/* Sprint table */}
      <Block>
        <p className="text-xs font-black uppercase tracking-widest text-gray-400 mb-4">
          🔢 Sprint Breakdown
        </p>
        <div className="overflow-x-auto -mx-1">
          <table className="w-full text-xs min-w-96">
            <thead>
              <tr className="border-b-2 border-gray-100">
                {["#","LEG","QCA","TAI","Net","Cumulative"].map(h=>(
                  <th key={h} className="py-2 px-2 font-black text-gray-500 text-right first:text-left">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r,i) => (
                <tr key={i} className="border-b border-gray-50 hover:bg-gray-50 transition-colors">
                  <td className="py-2 px-2 font-bold text-gray-700">S{r.s}</td>
                  <td className="py-2 px-2 text-right font-semibold text-emerald-700">{$(r.leg)}</td>
                  <td className="py-2 px-2 text-right font-semibold text-amber-700">{$(r.qca)}</td>
                  <td className="py-2 px-2 text-right font-semibold text-red-500">({$(r.tai)})</td>
                  <td className={`py-2 px-2 text-right font-bold ${r.net>=0?"text-emerald-700":"text-red-600"}`}>{$(r.net)}</td>
                  <td className={`py-2 px-2 text-right font-bold ${r.cum>=0?"text-indigo-700":"text-red-500"}`}>{$(r.cum)}</td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr className="border-t-2 border-gray-200 bg-gray-50 font-black">
                <td className="py-2.5 px-2 text-gray-800">Total</td>
                <td className="py-2.5 px-2 text-right text-emerald-800">{$(tL)}</td>
                <td className="py-2.5 px-2 text-right text-amber-800">{$(tQ)}</td>
                <td className="py-2.5 px-2 text-right text-red-700">({$(tT)})</td>
                <td className="py-2.5 px-2 text-right text-gray-900" colSpan={2}>{$(cum)}</td>
              </tr>
            </tfoot>
          </table>
        </div>
      </Block>

      {/* Insights */}
      <div className="rounded-2xl bg-slate-900 p-5 border border-slate-700 mb-4">
        <p className="text-xs font-black uppercase tracking-widest text-slate-400 mb-4">🔍 Key Insights</p>
        <ul className="space-y-2.5 text-sm text-slate-300">
          <li>• APM of <span className="text-white font-bold">{fix(apm)}×</span> — your team delivers{" "}
            <span className="text-emerald-400 font-bold">{pct((apm-1)*100, 1)}</span> more value per sprint with GenAI.</li>
          <li>• Labour savings drive <span className="text-white font-bold">{((tL/(tL+tQ))*100).toFixed(0)}%</span> of gross benefit;
            quality savings drive <span className="text-white font-bold">{((tQ/(tL+tQ))*100).toFixed(0)}%</span>.</li>
          <li>• AI tooling (TAI) costs only <span className="text-white font-bold">{((tT/(tL+tQ))*100).toFixed(0)}%</span> of gross savings — very efficient investment.</li>
          <li>• Technical debt predicted to fall <span className="text-amber-400 font-bold">{PRE_TDR}% → {fix(tdrPred,1)}%</span>{" "}
            (−{fix(PRE_TDR-tdrPred,1)} percentage points).</li>
          {breakEven
            ? <li>• <span className="text-emerald-400 font-bold">Breakeven Sprint {breakEven}</span> — setup cost fully recovered within the project.</li>
            : <li className="text-red-400">• Setup cost not recovered within the project. Increase sprint count or reduce setup cost.</li>}
          <li className={`text-xs pt-1 border-t border-slate-700 ${source==="ml"?"text-indigo-400":"text-amber-400"}`}>
            {source === "ml"
              ? `🧠 Results powered by the trained Random Forest ML model (R²=0.728, LOOCV-P, n=198 sprints).`
              : `📐 Results powered by the JS formula approximation. Connect the FastAPI backend to use the actual ML model.`}
          </li>
        </ul>
      </div>

      <p className="text-center text-xs text-gray-400 pb-8">
        NVAA Framework · Empirically calibrated on 198 real sprint observations<br/>
        Formula: NVAA = LEG + QCA − TAI
      </p>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════
//  ROOT APP
// ═══════════════════════════════════════════════════════════════
export default function App() {
  const [step,    setStep]    = useState(1)
  const [p1,      setP1]      = useState(D1)
  const [p2,      setP2]      = useState(D2)
  const [p3,      setP3]      = useState(D3)
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)

  const handleCalculate = async () => {
    setLoading(true)
    try {
      const r = await computeResult(p1, p2, p3)
      setResult(r)
      setStep(4)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setResult(null)
    setStep(1)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-800 to-violet-900 px-4 py-4 sticky top-0 z-20 shadow-lg">
        <div className="max-w-lg mx-auto flex items-center justify-between">
          <div>
            <p className="font-black text-white text-xl leading-none">
              NVAA <span className="font-normal text-indigo-200">Estimator</span>
            </p>
            <p className="text-indigo-300 text-xs mt-0.5">Net Value of AI Adoption</p>
          </div>
          <div className="text-right">
            {result?.source && (
              <span className={`text-xs font-bold px-2 py-1 rounded-full ${
                result.source === "ml" ? "bg-emerald-500 text-white" : "bg-amber-400 text-amber-900"}`}>
                {result.source === "ml" ? "🧠 ML Active" : "📐 Formula Mode"}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="max-w-lg mx-auto px-4 pt-6 pb-12">
        <Steps current={step} />
        {step === 1 && <Step1 d={p1} set={setP1} onNext={() => setStep(2)} />}
        {step === 2 && <Step2 d={p2} proj={p1} set={setP2} onNext={() => setStep(3)} onBack={() => setStep(1)} />}
        {step === 3 && <Step3 d={p3} set={setP3} onNext={handleCalculate} onBack={() => setStep(2)} loading={loading} />}
        {step === 4 && <Results result={result} proj={p1} ai={p2} fin={p3} onReset={handleReset} />}
      </div>
    </div>
  )
}
