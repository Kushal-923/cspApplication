// src/pages/Dashboard.tsx
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, SimulationStatus } from '../api/client'

function stagePill(stage: string) {
  const map: Record<string, { label: string; cls: string }> = {
    pending:          { label: 'Pending',    cls: 'pill-gray' },
    validating:       { label: 'Validating', cls: 'pill-blue' },
    placement:        { label: 'Placement',  cls: 'pill-blue' },
    coverage_matrix:  { label: 'Coverage',   cls: 'pill-blue' },
    scheduling:       { label: 'Scheduling', cls: 'pill-blue' },
    benchmarking:     { label: 'Benchmark',  cls: 'pill-blue' },
    czml_generation:  { label: 'CZML Gen',   cls: 'pill-blue' },
    completed:        { label: 'Complete',   cls: 'pill-green' },
    failed:           { label: 'Failed',     cls: 'pill-red' },
  }
  const info = map[stage] ?? { label: stage, cls: 'pill-gray' }
  return <span className={`pill ${info.cls}`}>{info.label}</span>
}

function elapsedStr(created: string, completed: string | null): string {
  const start = new Date(created).getTime()
  const end = completed ? new Date(completed).getTime() : Date.now()
  const secs = Math.floor((end - start) / 1000)
  if (secs < 60) return `${secs}s`
  const mins = Math.floor(secs / 60)
  return `${mins}m ${secs % 60}s`
}

export default function Dashboard() {
  const [simulations, setSimulations] = useState<SimulationStatus[]>([])
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  useEffect(() => {
    const load = () =>
      api.simulations.list().then(r => setSimulations(r.simulations)).finally(() => setLoading(false))
    load()
    const iv = setInterval(load, 5000)
    return () => clearInterval(iv)
  }, [])

  const running = simulations.filter(s =>
    !['completed','failed'].includes(s.current_stage)
  )

  return (
    <div>
      <div className="page-header">
        <div>
          <h2 className="page-title">Dashboard</h2>
          <p className="page-subtitle">Border Radar Optimization &amp; Scheduling Platform</p>
        </div>
        <button className="btn btn-primary" onClick={() => navigate('/simulations/new')}>
          ＋ New Simulation
        </button>
      </div>

      <div className="page-body">

        {/* Stats row */}
        <div className="metrics-grid" style={{ marginBottom: 24 }}>
          <div className="metric-card">
            <div className="metric-label">Total Runs</div>
            <div className="metric-value">{simulations.length}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Running</div>
            <div className="metric-value" style={{ color: running.length ? 'var(--color-primary)' : undefined }}>
              {running.length}
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Completed</div>
            <div className="metric-value" style={{ color: 'var(--color-accent)' }}>
              {simulations.filter(s => s.current_stage === 'completed').length}
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Failed</div>
            <div className="metric-value" style={{ color: 'var(--color-error)' }}>
              {simulations.filter(s => s.current_stage === 'failed').length}
            </div>
          </div>
        </div>

        {/* System info card */}
        <div className="card" style={{ marginBottom: 24 }}>
          <div className="card-title">System Overview</div>
          <p style={{ fontSize: 13, color: 'var(--color-text-dim)', lineHeight: 1.7 }}>
            This platform orchestrates an end-to-end pipeline of geospatial and combinatorial 
            optimization algorithms: radar placement via CP-SAT (MCLP), terrain-aware 3D LOS/coverage 
            matrix construction with DEM ray marching and 4/3 earth curvature, two-phase ILP scheduling 
            with energy &amp; thermal cooling constraints and frequency assignment, benchmarking against 
            Always-ON / Random / Greedy baselines, and CZML-based 3D visualization in CesiumJS.
          </p>
          <div style={{ marginTop: 12, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {['OR-Tools CP-SAT','GLO-30 DEM','GeoJSON Borders','CesiumJS 3D','Frequency Hopping'].map(t => (
              <span key={t} className="meta-chip">{t}</span>
            ))}
          </div>
        </div>

        {/* Recent simulations */}
        <div className="card">
          <div className="card-title">Recent Simulations</div>
          {loading ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '12px 0', color: 'var(--color-text-muted)' }}>
              <span className="spinner" /> Loading...
            </div>
          ) : simulations.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">🛰</div>
              <div className="empty-title">No simulations yet</div>
              <div className="empty-desc">Click <strong>New Simulation</strong> to run the full optimization pipeline.</div>
            </div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Simulation ID</th>
                  <th>Region</th>
                  <th>Status</th>
                  <th>Elapsed</th>
                  <th>Started</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {simulations.slice().reverse().map(sim => (
                  <tr key={sim.simulation_id} style={{ cursor: 'pointer' }}
                    onClick={() => navigate(`/simulations/${sim.simulation_id}`)}>
                    <td style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11 }}>
                      {sim.simulation_id}
                    </td>
                    <td>{sim.region_name}</td>
                    <td>{stagePill(sim.current_stage)}</td>
                    <td style={{ fontVariantNumeric: 'tabular-nums', fontSize: 12 }}>
                      {elapsedStr(sim.created_at, sim.completed_at)}
                    </td>
                    <td style={{ fontSize: 12, color: 'var(--color-text-muted)' }}>
                      {new Date(sim.created_at).toLocaleString()}
                    </td>
                    <td>
                      {sim.current_stage === 'completed' && (
                        <button className="btn btn-ghost" style={{ padding: '4px 10px', fontSize: 11 }}
                          onClick={e => { e.stopPropagation(); navigate(`/simulations/${sim.simulation_id}/visualize`) }}>
                          3D View
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  )
}
