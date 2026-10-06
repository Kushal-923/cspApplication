// src/pages/NewSimulation.tsx
import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, Region, ValidationResult } from '../api/client'

const SAMPLE_CONFIG = {
  small: {
    count: 5,
    range_m: 20000,
    h_ant: 5.0,
    theta_min: -5.0,
    theta_max: 20.0,
    az_halfwidth: 60.0,
    energy_budget: 24.0,
    cooling_L: 5,
    cooling_C: 2,
    frequency_bands: ["VHF", "UHF"]
  },
  med: {
    count: 5,
    range_m: 1000000,
    h_ant: 5.0,
    theta_min: -5.0,
    theta_max: 20.0,
    az_halfwidth: 60.0,
    energy_budget: 24.0,
    cooling_L: 5,
    cooling_C: 2,
    frequency_bands: ["UHF", "L_BAND", "S_BAND"]
  }
}

type Step = 1 | 2 | 3

export default function NewSimulation() {
  const navigate = useNavigate()
  const [step, setStep] = useState<Step>(1)

  // Step 1
  const [regions, setRegions] = useState<Region[]>([])
  const [selectedRegion, setSelectedRegion] = useState<Region | null>(null)
  const [regionsLoading, setRegionsLoading] = useState(true)

  // Step 2
  const [configText, setConfigText] = useState(JSON.stringify(SAMPLE_CONFIG, null, 2))
  const [parseError, setParseError] = useState<string | null>(null)
  const [validation, setValidation] = useState<ValidationResult | null>(null)
  const [validating, setValidating] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Step 3
  const [launching, setLaunching] = useState(false)
  const [launchError, setLaunchError] = useState<string | null>(null)

  useEffect(() => {
    api.regions.list()
      .then(r => setRegions(r.regions))
      .catch(() => {})
      .finally(() => setRegionsLoading(false))
  }, [])

  // ── Parse + validate config ────────────────────────────────────────────────
  const getParsedConfig = useCallback(() => {
    try {
      return { ok: true, config: JSON.parse(configText) }
    } catch (e: any) {
      return { ok: false, error: e.message }
    }
  }, [configText])

  const handleValidate = async () => {
    const parsed = getParsedConfig()
    if (!parsed.ok) {
      setParseError(parsed.error!)
      setValidation(null)
      return
    }
    setParseError(null)
    setValidating(true)
    try {
      const result = await api.simulations.validateConfig(parsed.config)
      setValidation(result)
    } catch {
      setValidation({ valid: false, errors: ['Server validation failed.'] })
    } finally {
      setValidating(false)
    }
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = ev => {
      const text = ev.target?.result as string
      setConfigText(text)
      setValidation(null)
      setParseError(null)
    }
    reader.readAsText(file)
  }

  // ── Launch ─────────────────────────────────────────────────────────────────
  const handleLaunch = async () => {
    if (!selectedRegion || !validation?.valid) return
    const parsed = getParsedConfig()
    if (!parsed.ok) return

    setLaunching(true)
    setLaunchError(null)
    try {
      const result = await api.simulations.create(selectedRegion.id, parsed.config)
      navigate(`/simulations/${result.simulation_id}`)
    } catch (err: any) {
      const detail = err?.response?.data?.detail
      if (typeof detail === 'string') setLaunchError(detail)
      else if (Array.isArray(detail)) setLaunchError(detail.map((d: any) => d.msg || d).join('; '))
      else if (detail?.message) setLaunchError(`${detail.message}: ${detail.errors?.join(', ')}`)
      else setLaunchError('Failed to launch simulation. Check server logs.')
    } finally {
      setLaunching(false)
    }
  }

  const canProceedStep1 = !!selectedRegion && selectedRegion.geojson_available && selectedRegion.dem_available
  const canProceedStep2 = validation?.valid === true
  const canLaunch = canProceedStep1 && canProceedStep2

  return (
    <div>
      <div className="page-header">
        <div>
          <h2 className="page-title">New Simulation</h2>
          <p className="page-subtitle">Configure and launch the full optimization pipeline</p>
        </div>
      </div>
      <div className="page-body">

        {/* Step tabs */}
        <div className="step-tabs">
          {([
            [1, 'Select Region'],
            [2, 'Radar Configuration'],
            [3, 'Review & Launch'],
          ] as [Step, string][]).map(([n, label]) => (
            <div
              key={n}
              className={`step-tab${step === n ? ' active' : ''}${step > n ? ' done' : ''}`}
              onClick={() => { if (n < step || (n === 2 && canProceedStep1) || (n === 3 && canProceedStep2)) setStep(n) }}
            >
              <span className="step-number">{step > n ? '✓' : n}</span>
              {label}
            </div>
          ))}
        </div>

        {/* ── Step 1: Select Region ──────────────────────────────────────────── */}
        {step === 1 && (
          <div>
            <p style={{ fontSize: 13, color: 'var(--color-text-muted)', marginBottom: 18 }}>
              Select the geographic region for this simulation. The corresponding GeoJSON border
              file and DEM will be loaded automatically.
            </p>
            {regionsLoading ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: 'var(--color-text-muted)' }}>
                <span className="spinner" /> Loading regions...
              </div>
            ) : regions.length === 0 ? (
              <div className="alert alert-info">No regions found. Add a region to data/regions/.</div>
            ) : (
              <div className="region-grid">
                {regions.map(r => (
                  <div
                    key={r.id}
                    className={`region-card${selectedRegion?.id === r.id ? ' selected' : ''}`}
                    onClick={() => setSelectedRegion(r)}
                  >
                    <div className="region-name">{r.name}</div>
                    <div className="region-desc">{r.description}</div>
                    <div className="region-meta">
                      <span className={`meta-chip`} style={{ color: r.geojson_available ? 'var(--color-accent)' : 'var(--color-error)' }}>
                        {r.geojson_available ? '✓' : '✗'} GeoJSON
                      </span>
                      <span className={`meta-chip`} style={{ color: r.dem_available ? 'var(--color-accent)' : 'var(--color-error)' }}>
                        {r.dem_available ? '✓' : '✗'} DEM
                      </span>
                      {r.resolution_m && <span className="meta-chip">{r.resolution_m}m res</span>}
                      {r.geojson_size_bytes && (
                        <span className="meta-chip">{Math.round(r.geojson_size_bytes / 1024)} KB GeoJSON</span>
                      )}
                    </div>
                    {!r.dem_available && (
                      <div className="alert alert-error" style={{ marginTop: 10, marginBottom: 0 }}>
                        DEM file missing. Place dem.tif in data/regions/{r.id}/
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
            <div style={{ marginTop: 24 }}>
              <button className="btn btn-primary btn-lg" disabled={!canProceedStep1}
                onClick={() => setStep(2)}>
                Continue →
              </button>
              {selectedRegion && !selectedRegion.dem_available && (
                <p style={{ marginTop: 10, fontSize: 12, color: 'var(--color-error)' }}>
                  DEM file is required. See README for download instructions.
                </p>
              )}
            </div>
          </div>
        )}

        {/* ── Step 2: Radar Configuration ───────────────────────────────────── */}
        {step === 2 && (
          <div>
            <p style={{ fontSize: 13, color: 'var(--color-text-muted)', marginBottom: 18 }}>
              Provide the radar type configuration JSON. This defines radar types, counts, 
              hardware parameters, energy constraints, and frequency bands.
            </p>

            {/* Upload */}
            <div className="card" style={{ marginBottom: 18 }}>
              <div className="card-title">Upload Config File</div>
              <label className="upload-zone">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".json"
                  onChange={handleFileUpload}
                  style={{ position: 'absolute', inset: 0, opacity: 0, cursor: 'pointer' }}
                />
                <div className="upload-icon">📁</div>
                <div className="upload-text">Click or drag a JSON file here</div>
                <div className="upload-hint">radar_type_config.json</div>
              </label>
            </div>

            {/* Editor */}
            <div className="card" style={{ marginBottom: 18 }}>
              <div className="card-title">Edit Configuration</div>
              {parseError && (
                <div className="alert alert-error" style={{ marginBottom: 10 }}>⚠ JSON parse error: {parseError}</div>
              )}
              <textarea
                className="form-textarea"
                value={configText}
                onChange={e => { setConfigText(e.target.value); setValidation(null); setParseError(null) }}
                style={{ minHeight: 320 }}
                spellCheck={false}
              />
            </div>

            {/* Validation result */}
            {validation && (
              <div className={`alert ${validation.valid ? 'alert-success' : 'alert-error'}`} style={{ marginBottom: 18 }}>
                {validation.valid ? (
                  <div>
                    <strong>✓ Configuration valid</strong>
                    {validation.summary && (
                      <ul style={{ marginTop: 6, paddingLeft: 16, fontSize: 12 }}>
                        <li>{validation.summary.total_radars} total radars across {validation.summary.num_types} type(s)</li>
                        <li>Types: {validation.summary.types.join(', ')}</li>
                        <li>Frequency bands: {validation.summary.frequency_bands.join(', ')}</li>
                      </ul>
                    )}
                  </div>
                ) : (
                  <div>
                    <strong>✗ Validation failed</strong>
                    <ul style={{ marginTop: 6, paddingLeft: 16, fontSize: 12 }}>
                      {validation.errors.map((e, i) => <li key={i}>{e}</li>)}
                    </ul>
                  </div>
                )}
              </div>
            )}

            <div style={{ display: 'flex', gap: 10 }}>
              <button className="btn btn-ghost" onClick={() => setStep(1)}>← Back</button>
              <button className="btn btn-ghost" onClick={handleValidate} disabled={validating}>
                {validating ? <><span className="spinner" /> Validating...</> : 'Validate Config'}
              </button>
              <button className="btn btn-primary btn-lg" disabled={!canProceedStep2}
                onClick={() => setStep(3)}>
                Continue →
              </button>
            </div>
          </div>
        )}

        {/* ── Step 3: Review & Launch ────────────────────────────────────────── */}
        {step === 3 && (
          <div>
            <div className="card" style={{ marginBottom: 18 }}>
              <div className="card-title">Simulation Summary</div>
              <table className="data-table">
                <tbody>
                  <tr><td style={{ color: 'var(--color-text-muted)', width: 160 }}>Region</td>
                    <td><strong>{selectedRegion?.name}</strong></td></tr>
                  <tr><td style={{ color: 'var(--color-text-muted)' }}>GeoJSON</td>
                    <td><span className="pill pill-green">✓ Loaded</span></td></tr>
                  <tr><td style={{ color: 'var(--color-text-muted)' }}>DEM</td>
                    <td><span className="pill pill-green">✓ Loaded</span></td></tr>
                  {validation?.summary && <>
                    <tr><td style={{ color: 'var(--color-text-muted)' }}>Radar Types</td>
                      <td>{validation.summary.num_types}</td></tr>
                    <tr><td style={{ color: 'var(--color-text-muted)' }}>Total Radars</td>
                      <td>{validation.summary.total_radars}</td></tr>
                    <tr><td style={{ color: 'var(--color-text-muted)' }}>Frequency Bands</td>
                      <td>{validation.summary.frequency_bands.join(', ')}</td></tr>
                  </>}
                  <tr><td style={{ color: 'var(--color-text-muted)' }}>Config</td>
                    <td><span className="pill pill-green">✓ Valid</span></td></tr>
                </tbody>
              </table>
            </div>

            <div className="card" style={{ marginBottom: 18, background: 'rgba(59,130,246,0.04)', borderColor: 'rgba(59,130,246,0.2)' }}>
              <div className="card-title">Pipeline Stages</div>
              <div className="stage-list">
                {[
                  'Input Validation',
                  'Radar Placement Optimization (CP-SAT MCLP)',
                  'LOS / Coverage Matrix (DEM Ray Marching)',
                  'CSP / ILP Scheduling (Two-Phase)',
                  'Benchmarking (B1/B2/B3 vs CSP)',
                  'CZML Generation',
                ].map((label, i) => (
                  <div key={i} className="stage-row">
                    <span className="stage-icon waiting" style={{ background: 'rgba(59,130,246,0.1)', color: 'var(--color-primary)' }}>
                      {i + 1}
                    </span>
                    <span className="stage-label">{label}</span>
                    <span style={{ fontSize: 11, color: 'var(--color-text-muted)' }}>Queued</span>
                  </div>
                ))}
              </div>
            </div>

            {launchError && (
              <div className="alert alert-error">{launchError}</div>
            )}

            <div style={{ display: 'flex', gap: 10 }}>
              <button className="btn btn-ghost" onClick={() => setStep(2)}>← Back</button>
              <button
                className="btn btn-accent btn-lg"
                disabled={!canLaunch || launching}
                onClick={handleLaunch}
              >
                {launching
                  ? <><span className="spinner" /> Launching...</>
                  : '▶ Start Simulation'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
