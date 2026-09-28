import { useState, useMemo } from "react"
import {
  BarChart, Bar, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine,
} from "recharts"

// ══════════════════════════════════════════════════════════════════
//  CONSTANTS  — calibrated from 198 real sprint observations
// ══════════════════════════════════════════════════════════════════
const DOMAIN_APM = {
  FinTech:1.072, Healthcare:1.046, Manufacturing:1.094,
  Telecom:1.009, Retail:1.050, Other:1.061,
}
const DOMAINS  = Object.keys(DOMAIN_APM)
const PRE_DD   = 0.127    // defects / SP  (pre-GenAI baseline)
const PRE_TDR  = 20.50    // tech debt %   (pre-GenAI baseline)
const MEAN_APM = 1.061

// ══════════════════════════════════════════════════════════════════
//  NVAA FORMULA ENGINE  (JS port — no backend required)
// ══════════════════════════════════════════════════════════════════
function getAPM({ intensity, maturity, complexity, numDevs, domain }) {
  const base = DOMAIN_APM[domain] ?? MEAN_APM
  return Math.max(0.98, Math.min(2.0,
    base
    + (intensity - 0.567) * 0.10     // usage intensity effect
    + (maturity  - 1.700) * 0.015    // AI maturity effect
    + (3.0 - complexity)  * 0.008    // complexity drag
    - Math.max(0, numDevs - 5) * 0.004  // Brooks' Law overhead
  ))
}

function runProject({ numDevs, sprintDays, totalSprints, domain, complexity,
                      intensity, maturity, laborRate, defectCost, aiCost, setupCost }) {
  const apm     = getAPM({ intensity, maturity, complexity, numDevs, domain })
  const ddPred  = Math.max(0, PRE_DD  - intensity * 0.0159)
  const tdrPred = Math.max(0, PRE_TDR - intensity * 9.171)
  const vel     = numDevs * sprintDays * 0.778 * apm   // SP/sprint estimate

  const rows = []
  let cum = -setupCost

  for (let s = 1; s <= totalSprints; s++) {
    const E   = numDevs * sprintDays * 8                      // person-hours
    const leg = E * (1 - 1 / apm) * laborRate
    const qca = Math.max(0, PRE_DD  - ddPred)  * vel * defectCost
              + Math.max(0, PRE_TDR - tdrPred) * 0.01 * E * laborRate * 0.30
    const tai = numDevs * sprintDays * aiCost
              + E * intensity * 0.08 * laborRate
    const disc = (leg + qca - tai) / Math.pow(1.001, s * sprintDays / 7)
    cum += disc
    rows.push({
      s,
      leg: +leg.toFixed(0), qca: +qca.toFixed(0),
      tai: +tai.toFixed(0), net: +(leg+qca-tai).toFixed(0),
      cum: +cum.toFixed(0),
    })
  }

  const tLEG = rows.reduce((a,r) => a+r.leg, 0)
  const tQCA = rows.reduce((a,r) => a+r.qca, 0)
  const tTAI = rows.reduce((a,r) => a+r.tai, 0)
  const be   = rows.findIndex(r => r.cum >= 0)  // 0-indexed sprint index

  return {
    apm, ddPred, tdrPred, rows, cum,
    tLEG, tQCA, tTAI,
    roi:       cum / (tTAI + setupCost) * 100,
    effortSav: rows.reduce((a,r) => a + r.leg / laborRate, 0),
    breakEven: be >= 0 ? be + 1 : null,
  }
}

// ══════════════════════════════════════════════════════════════════
//  FORMATTING
// ══════════════════════════════════════════════════════════════════
const $   = n => new Intl.NumberFormat("en-US",{style:"currency",currency:"USD",maximumFractionDigits:0}).format(n)
const fix = (n,d=3) => n.toFixed(d)
const pct = (n,d=1) => `${n>=0?"+":""}${n.toFixed(d)}%`

// ══════════════════════════════════════════════════════════════════
//  DEFAULT FORM VALUES
// ══════════════════════════════════════════════════════════════════
const D1 = { numDevs:5, totalSprints:10, sprintDays:10, domain:"FinTech", complexity:3 }
const D2 = { enabled:true, intensity:0.65, maturity:2 }
const D3 = { laborRate:75, defectCost:2000, aiCost:0.50, setupCost:3000 }

// ══════════════════════════════════════════════════════════════════
//  REUSABLE UI ATOMS
// ══════════════════════════════════════════════════════════════════
function Pill({ label, active, color="blue", onClick }) {
  const base  = "flex-1 py-2.5 px-3 rounded-xl text-sm font-semibold border transition-all cursor-pointer select-none text-center"
  const on    = color==="green" ? "bg-green-600 text-white border-green-600"
              : color==="red"   ? "bg-red-100 text-red-700 border-red-400"
              : "bg-blue-600 text-white border-blue-600"
  const off   = "bg-white text-gray-600 border-gray-200 hover:border-blue-300"
  return <button className={`${base} ${active?on:off}`} onClick={onClick}>{label}</button>
}

function Slider({ value, min, max, step=1, onChange, display }) {
  return (
    <div className="flex items-center gap-3">
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(+e.target.value)}
        className="flex-1 accent-blue-600 cursor-pointer" />
      <span className="w-20 text-right text-sm font-bold text-blue-700 shrink-0">
        {display(value)}
      </span>
    </div>
  )
}

function Field({ label, hint, children }) {
  return (
    <div className="mb-5">
      <p className="text-sm font-semibold text-gray-800 mb-0.5">{label}</p>
      {hint && <p className="text-xs text-gray-400 mb-2">{hint}</p>}
      {children}
    </div>
  )
}

function Card({ children, className="" }) {
  return (
    <div className={`bg-white rounded-2xl border border-gray-100 shadow-sm p-5 mb-4 ${className}`}>
      {children}
    </div>
  )
}

function SectionTitle({ icon, text }) {
  return (
    <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3 flex items-center gap-1.5">
      <span>{icon}</span>{text}
    </p>
  )
}

function NavRow({ onBack, onNext, nextLabel="Next →", disabled=false }) {
  return (
    <div className="flex gap-3 mt-6">
      {onBack && (
        <button onClick={onBack}
          className="flex-1 py-3 rounded-xl border border-gray-300 text-gray-600 font-semibold hover:bg-gray-50 transition-colors">
          ← Back
        </button>
      )}
      <button onClick={onNext} disabled={disabled}
        className="flex-1 py-3 rounded-xl bg-blue-600 text-white font-semibold hover:bg-blue-700 disabled:opacity-40 transition-colors">
        {nextLabel}
      </button>
    </div>
  )
}

// ══════════════════════════════════════════════════════════════════
//  PROGRESS BAR
// ══════════════════════════════════════════════════════════════════
function ProgressBar({ step }) {
  const steps = ["Project","AI Config","Financial","Results"]
  return (
    <div className="flex items-center justify-center mb-8 gap-1">
      {steps.map((s,i) => (
        <div key={i} className="flex items-center">
          <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold transition-all
            ${i+1 < step  ? "bg-blue-600 text-white"
            : i+1===step  ? "bg-blue-600 text-white ring-4 ring-blue-100"
            :               "bg-gray-100 text-gray-400"}`}>
            {i+1 < step ? "✓" : i+1}
          </div>
          <span className={`ml-1.5 text-xs font-medium hidden sm:inline
            ${i+1<=step ? "text-blue-600" : "text-gray-400"}`}>{s}</span>
          {i<steps.length-1 && (
            <div className={`w-6 h-0.5 mx-1.5 sm:mx-2 rounded ${i+1<step?"bg-blue-500":"bg-gray-200"}`}/>
          )}
        </div>
      ))}
    </div>
  )
}

// ══════════════════════════════════════════════════════════════════
//  STEP 1 — Project Setup
// ══════════════════════════════════════════════════════════════════
function Step1({ d, set, onNext }) {
  return (
    <div>
      <h2 className="text-xl font-black text-gray-900 mb-0.5">Project Setup</h2>
      <p className="text-sm text-gray-500 mb-5">Define your team and project context.</p>

      <Card>
        <SectionTitle icon="👥" text="Team & Timeline" />
        <Field label="Number of Developers">
          <Slider value={d.numDevs} min={1} max={25} onChange={v=>set({...d,numDevs:v})}
            display={v=>`${v} devs`} />
        </Field>
        <Field label="Total Sprints" hint="How many sprints in the project?">
          <Slider value={d.totalSprints} min={2} max={30} onChange={v=>set({...d,totalSprints:v})}
            display={v=>`${v} sprints`} />
        </Field>
        <Field label="Sprint Length" hint="Working days per sprint">
          <Slider value={d.sprintDays} min={5} max={30} step={5} onChange={v=>set({...d,sprintDays:v})}
            display={v=>`${v} days`} />
        </Field>
      </Card>

      <Card>
        <SectionTitle icon="🏢" text="Context" />
        <Field label="Industry Domain">
          <div className="grid grid-cols-3 gap-2">
            {DOMAINS.map(dm => (
              <Pill key={dm} label={dm} active={d.domain===dm}
                onClick={()=>set({...d,domain:dm})} />
            ))}
          </div>
        </Field>
        <Field label="Project Complexity" hint="1 = Simple CRUD  ·  5 = ML / distributed system">
          <Slider value={d.complexity} min={1} max={5} step={0.5}
            onChange={v=>set({...d,complexity:v})}
            display={v=>["","▪","▪▪","▪▪▪","▪▪▪▪","▪▪▪▪▪"][Math.round(v)]||v} />
        </Field>
      </Card>

      <NavRow onNext={onNext} />
    </div>
  )
}

// ══════════════════════════════════════════════════════════════════
//  STEP 2 — AI Configuration
// ══════════════════════════════════════════════════════════════════
function Step2({ d, proj, set, onNext, onBack }) {
  const previewAPM = useMemo(
    () => getAPM({ intensity: d.intensity, maturity: d.maturity,
                   complexity: proj.complexity, numDevs: proj.numDevs,
                   domain: proj.domain }),
    [d, proj]
  )

  return (
    <div>
      <h2 className="text-xl font-black text-gray-900 mb-0.5">AI Configuration</h2>
      <p className="text-sm text-gray-500 mb-5">How is your team adopting GenAI tooling?</p>

      <Card>
        <SectionTitle icon="🤖" text="Adoption" />
        <Field label="GenAI Enabled?">
          <div className="flex gap-2">
            <Pill label="✅  Enabled"  active={d.enabled}  color="green" onClick={()=>set({...d,enabled:true})}/>
            <Pill label="❌  Disabled" active={!d.enabled} color="red"   onClick={()=>set({...d,enabled:false})}/>
          </div>
        </Field>

        {d.enabled && (
          <>
            <Field label="Usage Intensity"
              hint="Fraction of tasks assisted by GenAI (0 = none, 1 = all tasks)">
              <Slider value={d.intensity} min={0} max={1} step={0.05}
                onChange={v=>set({...d,intensity:+v.toFixed(2)})}
                display={v=>`${(v*100).toFixed(0)}%`} />
            </Field>
            <Field label="Team AI Maturity">
              <div className="flex gap-2">
                {[[1,"Beginner"],[2,"Intermediate"],[3,"Expert"]].map(([m,lbl])=>(
                  <Pill key={m} label={<><span className="block text-base">L{m}</span><span className="text-xs opacity-60">{lbl}</span></>}
                    active={d.maturity===m} onClick={()=>set({...d,maturity:m})} />
                ))}
              </div>
            </Field>

            <div className="mt-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-4 border border-blue-100">
              <p className="text-xs font-semibold text-blue-700 uppercase tracking-wide mb-1">
                Live APM Preview
              </p>
              <div className="flex items-end gap-3">
                <span className="text-3xl font-black text-blue-800">{fix(previewAPM,3)}×</span>
                <span className="text-sm text-blue-600 pb-1">
                  ≈ {pct((previewAPM-1)*100,1)} productivity uplift
                </span>
              </div>
              <p className="text-xs text-blue-500 mt-1">
                AI Productivity Multiplier — SHAP-calibrated from Random Forest (R²=0.728, n=198)
              </p>
            </div>
          </>
        )}
      </Card>

      <NavRow onBack={onBack} onNext={onNext} disabled={d.enabled && d.intensity===0} />
    </div>
  )
}

// ══════════════════════════════════════════════════════════════════
//  STEP 3 — Financial Parameters
// ══════════════════════════════════════════════════════════════════
function Step3({ d, set, onNext, onBack }) {
  return (
    <div>
      <h2 className="text-xl font-black text-gray-900 mb-0.5">Financial Parameters</h2>
      <p className="text-sm text-gray-500 mb-5">Calibrate cost inputs to your organisation.</p>

      <Card>
        <SectionTitle icon="💰" text="Labour & Quality Costs" />
        <Field label="Fully-Loaded Labour Rate ($/hour)"
          hint="Include salary, benefits, overhead — typically $60–$150/hr">
          <Slider value={d.laborRate} min={20} max={200} step={5}
            onChange={v=>set({...d,laborRate:v})} display={v=>`$${v}/hr`} />
        </Field>
        <Field label="Avg. Defect Fix Cost ($)"
          hint="Cost to remediate one escaped production defect — Jones (2012): $500–$20,000">
          <Slider value={d.defectCost} min={500} max={15000} step={500}
            onChange={v=>set({...d,defectCost:v})}
            display={v=>`$${v.toLocaleString()}`} />
        </Field>
      </Card>

      <Card>
        <SectionTitle icon="⚙️" text="AI Tooling Costs" />
        <Field label="AI Tool Cost ($/developer/day)"
          hint="GitHub Copilot ≈ $0.45 · ChatGPT Plus ≈ $0.65 · Enterprise ≈ $1.30">
          <Slider value={d.aiCost} min={0.1} max={5} step={0.05}
            onChange={v=>set({...d,aiCost:+v.toFixed(2)})}
            display={v=>`$${v.toFixed(2)}`} />
        </Field>
        <Field label="One-Time Setup Cost ($)"
          hint="Licences · integration · onboarding · training">
          <Slider value={d.setupCost} min={0} max={20000} step={500}
            onChange={v=>set({...d,setupCost:v})}
            display={v=>`$${v.toLocaleString()}`} />
        </Field>
      </Card>

      <NavRow onBack={onBack} nextLabel="Calculate NVAA →" onNext={onNext} />
    </div>
  )
}

// ══════════════════════════════════════════════════════════════════
//  RESULTS DASHBOARD
// ══════════════════════════════════════════════════════════════════
function KPI({ label, value, sub, accent="blue" }) {
  const ring = {blue:"border-blue-400 bg-blue-50",green:"border-green-400 bg-green-50",
    amber:"border-amber-400 bg-amber-50",red:"border-red-400 bg-red-50"}
  const txt  = {blue:"text-blue-800",green:"text-green-800",amber:"text-amber-800",red:"text-red-800"}
  return (
    <div className={`rounded-xl border-2 p-3 ${ring[accent]}`}>
      <p className={`text-xs font-bold uppercase tracking-wide opacity-60 ${txt[accent]}`}>{label}</p>
      <p className={`text-2xl font-black mt-0.5 ${txt[accent]}`}>{value}</p>
      {sub && <p className={`text-xs mt-0.5 opacity-70 ${txt[accent]}`}>{sub}</p>}
    </div>
  )
}

function TTip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-lg p-3 text-xs">
      <p className="font-bold mb-1 text-gray-700">Sprint {label}</p>
      {payload.map((p,i) => (
        <p key={i} style={{color:p.color}} className="font-medium">{p.name}: {$(p.value)}</p>
      ))}
    </div>
  )
}

function Results({ proj, ai, fin, onReset }) {
  const params = { ...proj, ...ai, ...fin }
  const r      = useMemo(() => ai.enabled ? runProject(params) : null, [])

  if (!ai.enabled) return (
    <div className="text-center py-16">
      <p className="text-5xl mb-3">🤖</p>
      <h3 className="text-lg font-bold text-gray-700">GenAI Disabled</h3>
      <p className="text-sm text-gray-400 mt-1 mb-6">Enable AI tools in Step 2 to compute NVAA.</p>
      <button onClick={onReset}
        className="px-8 py-3 bg-blue-600 text-white rounded-xl font-semibold">← Reconfigure</button>
    </div>
  )

  const isPos = r.cum >= 0

  const barData = [
    { name:"LEG",     value:+r.tLEG.toFixed(0), fill:"#16A34A" },
    { name:"QCA",     value:+r.tQCA.toFixed(0), fill:"#D97706" },
    { name:"TAI",     value:+r.tTAI.toFixed(0), fill:"#DC2626" },
    { name:"Net NVAA",value:+r.cum.toFixed(0),  fill: isPos?"#2563EB":"#DC2626" },
  ]

  const perSprintBar = r.rows.map(row => ({
    sprint: `S${row.s}`,
    LEG: row.leg, QCA: row.qca, TAI: -row.tai,
  }))

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-xl font-black text-gray-900">Results</h2>
          <p className="text-xs text-gray-400">{proj.numDevs} devs · {proj.totalSprints} sprints · {proj.domain}</p>
        </div>
        <button onClick={onReset}
          className="text-xs text-blue-600 font-semibold hover:underline">← Edit</button>
      </div>

      {/* Hero NVAA */}
      <div className={`rounded-2xl p-6 mb-4 text-center shadow-lg text-white
        ${isPos ? "bg-gradient-to-br from-emerald-500 to-green-700"
                : "bg-gradient-to-br from-red-500 to-rose-700"}`}>
        <p className="text-xs font-bold uppercase tracking-widest opacity-75">Cumulative NVAA</p>
        <p className="text-5xl font-black mt-1">{$(r.cum)}</p>
        <p className="text-xs opacity-75 mt-2">
          over {proj.totalSprints} sprints · ROI {fix(r.roi,1)}% ·
          {r.breakEven ? ` breakeven Sprint ${r.breakEven}` : " no breakeven in project"}
        </p>
      </div>

      {/* KPI grid */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <KPI label="AI Productivity Multiplier" value={`${fix(r.apm)}×`}
          sub={`${pct((r.apm-1)*100,1)} vs no-AI baseline`} accent="blue" />
        <KPI label="Return on Investment" value={`${fix(r.roi,1)}%`}
          sub="net / (TAI + setup)" accent={r.roi>=0?"green":"red"} />
        <KPI label="Labour Savings (LEG)" value={$(r.tLEG)}
          sub={`${fix(r.effortSav,0)} person-hours recovered`} accent="green" />
        <KPI label="Quality Savings (QCA)" value={$(r.tQCA)}
          sub={`TDR ${PRE_TDR}% → ${fix(r.tdrPred,1)}%`} accent="amber" />
      </div>

      {/* Total breakdown bar */}
      <Card>
        <p className="text-xs font-bold text-gray-500 uppercase tracking-wide mb-3">
          💰 Total Project Breakdown
        </p>
        <ResponsiveContainer width="100%" height={170}>
          <BarChart data={barData} layout="vertical" margin={{left:10,right:55}}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" tickFormatter={v=>`$${(v/1000).toFixed(0)}k`} tick={{fontSize:10}} />
            <YAxis type="category" dataKey="name" tick={{fontSize:11,fontWeight:600}} width={70}/>
            <Tooltip formatter={v=>[$(v)]} />
            <Bar dataKey="value" radius={[0,6,6,0]}
              label={{position:"right",formatter:v=>$(v),fontSize:10,fontWeight:600}}>
              {barData.map((d,i)=><Cell key={i} fill={d.fill}/>)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Per-sprint stacked bar */}
      <Card>
        <p className="text-xs font-bold text-gray-500 uppercase tracking-wide mb-3">
          📊 Per-Sprint Components
        </p>
        <ResponsiveContainer width="100%" height={160}>
          <BarChart data={perSprintBar} margin={{right:10}}>
            <CartesianGrid strokeDasharray="3 3" vertical={false}/>
            <XAxis dataKey="sprint" tick={{fontSize:9}}/>
            <YAxis tickFormatter={v=>`$${(v/1000).toFixed(0)}k`} tick={{fontSize:9}}/>
            <Tooltip content={<TTip/>}/>
            <Bar dataKey="LEG" stackId="a" fill="#16A34A" name="LEG"/>
            <Bar dataKey="QCA" stackId="a" fill="#D97706" name="QCA"/>
            <Bar dataKey="TAI" stackId="a" fill="#DC2626" name="TAI (cost)"/>
          </BarChart>
        </ResponsiveContainer>
        <div className="flex gap-4 justify-center mt-2 text-xs">
          {[["LEG","#16A34A","Labour Savings"],["QCA","#D97706","Quality Savings"],["TAI","#DC2626","AI Cost"]].map(([k,c,l])=>(
            <div key={k} className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded-sm inline-block" style={{background:c}}/>
              <span className="text-gray-500">{l}</span>
            </div>
          ))}
        </div>
      </Card>

      {/* Cumulative NVAA line */}
      <Card>
        <p className="text-xs font-bold text-gray-500 uppercase tracking-wide mb-3">
          📈 Cumulative NVAA Over Project
        </p>
        <ResponsiveContainer width="100%" height={190}>
          <AreaChart data={r.rows.map(row=>({sprint:`S${row.s}`,NVAA:row.cum}))} margin={{right:10}}>
            <defs>
              <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1">
                <stop offset="10%"  stopColor={isPos?"#2563EB":"#DC2626"} stopOpacity={0.25}/>
                <stop offset="95%"  stopColor={isPos?"#2563EB":"#DC2626"} stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3"/>
            <XAxis dataKey="sprint" tick={{fontSize:10}}/>
            <YAxis tickFormatter={v=>`$${(v/1000).toFixed(0)}k`} tick={{fontSize:10}}/>
            <Tooltip formatter={v=>[$(v),"Cumulative NVAA"]} labelFormatter={l=>`Sprint ${l.replace("S","")}`}/>
            <ReferenceLine y={0} stroke="#6B7280" strokeDasharray="4 4"/>
            <Area type="monotone" dataKey="NVAA" stroke={isPos?"#2563EB":"#DC2626"}
              strokeWidth={2.5} fill="url(#g1)" dot={{r:3}} activeDot={{r:5}}/>
          </AreaChart>
        </ResponsiveContainer>
        {r.breakEven && (
          <p className="text-xs text-center text-gray-400 mt-1">
            💡 Setup cost recovered by Sprint {r.breakEven} · Terminal value {$(r.cum)}
          </p>
        )}
      </Card>

      {/* Sprint table */}
      <Card>
        <p className="text-xs font-bold text-gray-500 uppercase tracking-wide mb-3">
          🔢 Sprint-by-Sprint Breakdown
        </p>
        <div className="overflow-x-auto -mx-1">
          <table className="w-full text-xs min-w-[440px]">
            <thead>
              <tr className="border-b-2 border-gray-100">
                {["#","LEG","QCA","TAI","Net","Cumulative"].map(h=>(
                  <th key={h} className="text-right py-2 px-2 font-bold text-gray-500 first:text-left">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {r.rows.map((row,i)=>(
                <tr key={i} className={`border-b border-gray-50 hover:bg-gray-50
                  ${row.cum>=0?"":"bg-red-50/40"}`}>
                  <td className="py-1.5 px-2 font-semibold text-gray-700">S{row.s}</td>
                  <td className="text-right py-1.5 px-2 text-green-700">{$(row.leg)}</td>
                  <td className="text-right py-1.5 px-2 text-amber-700">{$(row.qca)}</td>
                  <td className="text-right py-1.5 px-2 text-red-600">({$(row.tai)})</td>
                  <td className={`text-right py-1.5 px-2 font-bold ${row.net>=0?"text-green-700":"text-red-600"}`}>{$(row.net)}</td>
                  <td className={`text-right py-1.5 px-2 font-bold ${row.cum>=0?"text-blue-700":"text-red-500"}`}>{$(row.cum)}</td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr className="border-t-2 border-gray-200 bg-gray-50">
                <td className="py-2 px-2 font-black text-gray-700">Total</td>
                <td className="text-right py-2 px-2 font-black text-green-700">{$(r.tLEG)}</td>
                <td className="text-right py-2 px-2 font-black text-amber-700">{$(r.tQCA)}</td>
                <td className="text-right py-2 px-2 font-black text-red-600">({$(r.tTAI)})</td>
                <td className="text-right py-2 px-2 font-black text-gray-900" colSpan={2}>{$(r.cum)}</td>
              </tr>
            </tfoot>
          </table>
        </div>
      </Card>

      {/* Insight panel */}
      <Card className="bg-gradient-to-br from-slate-800 to-slate-900 border-slate-700">
        <p className="text-xs font-bold uppercase tracking-wide text-slate-400 mb-3">🔍 Key Insights</p>
        <ul className="space-y-2 text-sm text-slate-300">
          <li>• APM of <span className="text-white font-bold">{fix(r.apm)}×</span> means your team delivers <span className="text-green-400 font-bold">{pct((r.apm-1)*100,1)}</span> more per sprint with GenAI.</li>
          <li>• Labour savings (LEG) drive <span className="text-white font-bold">{((r.tLEG/(r.tLEG+r.tQCA))*100).toFixed(0)}%</span> of gross benefit; quality savings (QCA) drive <span className="text-white font-bold">{((r.tQCA/(r.tLEG+r.tQCA))*100).toFixed(0)}%</span>.</li>
          <li>• AI tooling costs (TAI) are only <span className="text-white font-bold">{((r.tTAI/(r.tLEG+r.tQCA))*100).toFixed(0)}%</span> of gross savings — high efficiency investment.</li>
          <li>• Tech debt ratio predicted to drop <span className="text-amber-400 font-bold">{PRE_TDR}% → {fix(r.tdrPred,1)}%</span> ({fix(PRE_TDR-r.tdrPred,1)} pp reduction).</li>
          {r.breakEven ? <li>• <span className="text-green-400 font-bold">Breakeven at Sprint {r.breakEven}</span> — setup cost fully recovered within the project.</li>
            : <li className="text-red-400">• Setup cost not recovered within project. Extend timeline or reduce setup cost.</li>}
        </ul>
      </Card>

      <p className="text-center text-xs text-gray-400 mt-4 pb-6">
        NVAA Framework · Calibrated on 198 sprint observations · APM via Random Forest (R²=0.728, LOOCV-P)
        <br/>Formula: NVAA = LEG + QCA − TAI · <a href="https://github.com" className="underline">View Source</a>
      </p>
    </div>
  )
}

// ══════════════════════════════════════════════════════════════════
//  ROOT APP
// ══════════════════════════════════════════════════════════════════
export default function App() {
  const [step, setStep] = useState(1)
  const [p1, setP1] = useState(D1)   // project
  const [p2, setP2] = useState(D2)   // AI config
  const [p3, setP3] = useState(D3)   // financial

  return (
    <div className="min-h-screen bg-gray-50 font-sans">

      {/* Sticky header */}
      <div className="bg-white border-b border-gray-200 px-4 py-3 sticky top-0 z-20 shadow-sm">
        <div className="max-w-md mx-auto flex items-center justify-between">
          <div>
            <span className="font-black text-blue-600 text-lg">NVAA</span>
            <span className="font-black text-gray-900 text-lg"> Estimator</span>
            <p className="text-xs text-gray-400 leading-none">Net Value of AI Adoption</p>
          </div>
          <div className="text-right">
            <p className="text-xs font-bold text-gray-500">Step {Math.min(step,4)}/4</p>
            <p className="text-xs text-gray-400">{["","Project","AI","Financial","Results"][Math.min(step,4)]}</p>
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="max-w-md mx-auto px-4 pt-6 pb-10">
        <ProgressBar step={step} />
        {step===1 && <Step1 d={p1} set={setP1} onNext={()=>setStep(2)} />}
        {step===2 && <Step2 d={p2} proj={p1} set={setP2}
                       onNext={()=>setStep(3)} onBack={()=>setStep(1)} />}
        {step===3 && <Step3 d={p3} set={setP3}
                       onNext={()=>setStep(4)} onBack={()=>setStep(2)} />}
        {step===4 && <Results proj={p1} ai={p2} fin={p3} onReset={()=>setStep(1)} />}
      </div>
    </div>
  )
}
