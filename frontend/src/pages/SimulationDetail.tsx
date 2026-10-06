// src/pages/SimulationDetail.tsx
import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api, SimulationStatus, SimulationResults } from '../api/client'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell
} from 'recharts'

const STAGE_ICONS: Record<string, string> = {
  done: '✓', running: '▶', waiting: '○', failed: '✗', skipped: '—'
}

const COLORS_BY_METHOD = {
  'Always-ON':  '#ef4444',
  'Random':     '#f59e0b',
  'Greedy':     '#3b82f6',
  'CSP / ILP':  '#10b981',
}

function StatusSection({ status }: { status: SimulationStatus }) {
  const isRunning = !['completed','failed'].includes(status.current_stage)
  const pct = status.stages.filter(s => s.status === 'done').length / status.stages.length * 100

  return (
    <div className="card" style={{ marginBottom: 18 }}>
      <div className="card-title">Pipeline Progress</div>
      {/* Progress bar */}
      <div style={{ height: 4, background: 'var(--color-bg-elevated)', borderRadius: 2, marginBottom: 18, overflow: 'hidden' }}>
        <div style={{
          height: '100%',
          width: `${pct}%`,
          background: status.current_stage === 'failed'
            ? 'var(--color-error)'
            : 'linear-gradient(90deg, var(--color-primary), var(--color-accent))',
          transition: 'width 0.4s ease',
          borderRadius: 2,
        }} />
      </div>

      <div className="stage-list">
        {status.stages.map(stage => (
          <div key={stage.name} className="stage-row">
            <span className={`stage-icon ${stage.status}`}>
              {stage.status === 'running'
                ? <span className="spinner" style={{ width: 12, height: 12, borderWidth: 2 }} />
                : STAGE_ICONS[stage.status] || '○'}
            </span>
            <span className={`stage-label ${stage.status}`}>{stage.label}</span>
            {stage.status === 'running' && (
              <span className="pill pill-blue" style={{ marginLeft: 'auto' }}>
                <span className="pulse-dot" style={{ width: 6, height: 6 }} /> Running
              </span>
            )}
            {stage.status === 'done' && stage.completed_at && (
              <span style={{ fontSize: 11, color: 'var(--color-text-muted)', marginLeft: 'auto' }}>
                {stage.started_at && stage.completed_at
                  ? `${((new Date(stage.completed_at).getTime() - new Date(stage.started_at).getTime()) / 1000).toFixed(0)}s`
                  : ''}
              </span>
            )}
            {stage.status === 'failed' && stage.error && (
              <span style={{ fontSize: 11, color: 'var(--color-error)', marginLeft: 'auto', maxWidth: 200, textAlign: 'right' }}>
                {stage.error.slice(0, 80)}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function ResultsSection({ results }: { results: SimulationResults }) {
  return (
    <>
      {/* Key metrics */}
      <div className="card" style={{ marginBottom: 18 }}>
        <div className="card-title">Results Summary</div>
        <div className="metrics-grid">
          {results.num_radars !== undefined && (
            <div className="metric-card">
              <div className="metric-label">Radars Placed</div>
              <div className="metric-value">{results.num_radars}</div>
            </div>
          )}
          {results.num_boundary_points !== undefined && (
            <div className="metric-card">
              <div className="metric-label">Border Points</div>
              <div className="metric-value">{results.num_boundary_points.toLocaleString()}</div>
            </div>
          )}
          {results.coverage_pct_any !== undefined && (
            <div className="metric-card">
              <div className="metric-label">Coverage</div>
              <div className="metric-value">{results.coverage_pct_any.toFixed(1)}<span className="metric-unit">%</span></div>
            </div>
          )}
          {results.phase !== undefined && (
            <div className="metric-card">
              <div className="metric-label">Solver Phase</div>
              <div className="metric-value">Phase {results.phase}</div>
            </div>
          )}
          {results.void_pct !== undefined && (
            <div className="metric-card">
              <div className="metric-label">Void %</div>
              <div className="metric-value">{results.void_pct.toFixed(2)}<span className="metric-unit">%</span></div>
            </div>
          )}
          {results.total_on_slots !== undefined && (
            <div className="metric-card">
              <div className="metric-label">ON Slots</div>
              <div className="metric-value">{results.total_on_slots}</div>
            </div>
          )}
          {results.overlap_density !== undefined && (
            <div className="metric-card">
              <div className="metric-label">Overlap Density</div>
              <div className="metric-value">{results.overlap_density.toFixed(4)}</div>
            </div>
          )}
          {results.solver_status && (
            <div className="metric-card">
              <div className="metric-label">Solver Status</div>
              <div className="metric-value" style={{ fontSize: 16 }}>{results.solver_status}</div>
            </div>
          )}
        </div>

        {results.frequency_bands_used && (
          <div style={{ marginTop: 16 }}>
            <div style={{ fontSize: 12, color: 'var(--color-text-muted)', marginBottom: 8 }}>Frequency Bands Used</div>
            <div style={{ display: 'flex', gap: 8 }}>
              {results.frequency_bands_used.map(b => (
                <span key={b} className="pill pill-blue">{b}</span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Baseline comparison chart */}
      {results.baselines && (
        <div className="card" style={{ marginBottom: 18 }}>
          <div className="card-title">Benchmark Comparison — Void %</div>
          {(() => {
            const b = results.baselines
            const data = [
              { name: 'Always-ON',  value: b.B1_always_on?.void_pct ?? 0 },
              { name: 'Random',     value: b.B2_random?.void_pct_mean ?? 0 },
              { name: 'Greedy',     value: b.B3_greedy?.void_pct ?? 0 },
              { name: 'CSP / ILP',  value: b.CSP_scheduler?.void_pct ?? 0 },
            ]
            return (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={data} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#94a3b8' }} />
                  <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} unit="%" />
                  <Tooltip
                    contentStyle={{ background: 'var(--color-bg-card)', border: '1px solid var(--color-border)', borderRadius: 8, fontSize: 12 }}
                    formatter={(v: any) => [`${Number(v).toFixed(2)}%`, 'Void %']}
                  />
                  <Bar dataKey="value" radius={[4,4,0,0]}>
                    {data.map(d => (
                      <Cell key={d.name} fill={COLORS_BY_METHOD[d.name as keyof typeof COLORS_BY_METHOD] || '#3b82f6'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )
          })()}
          <p style={{ fontSize: 11, color: 'var(--color-text-muted)', marginTop: 8 }}>
            Lower void % is better. CSP/ILP should achieve lowest void among constraint-satisfying methods.
          </p>
        </div>
      )}

      {/* Output files */}
      {results.output_files && Object.keys(results.output_files).length > 0 && (
        <div className="card" style={{ marginBottom: 18 }}>
          <div className="card-title">Output Files</div>
          <table className="data-table">
            <thead><tr><th>File</th><th>Download</th></tr></thead>
            <tbody>
              {Object.entries(results.output_files).map(([name, url]) => (
                <tr key={name}>
                  <td style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11 }}>{name}</td>
                  <td>
                    <a href={url} download={name} className="btn btn-ghost" style={{ padding: '3px 10px', fontSize: 11 }}>
                      ↓ Download
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  )
}

export default function SimulationDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [status, setStatus] = useState<SimulationStatus | null>(null)
  const [results, setResults] = useState<SimulationResults | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [activeTab, setActiveTab] = useState<'progress'|'logs'|'results'>('progress')
  const logRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!id) return
    let cancelled = false

    const poll = async () => {
      try {
        const s = await api.simulations.status(id!)
        if (cancelled) return
        setStatus(s)

        const l = await api.simulations.logs(id!, 500)
        if (cancelled) return
        setLogs(l.logs)

        if (s.current_stage === 'completed') {
          try {
            const r = await api.simulations.results(id!)
            if (!cancelled) setResults(r)
          } catch {}
        }
      } catch {}
    }

    poll()
    const iv = setInterval(poll, 2500)
    return () => { cancelled = true; clearInterval(iv) }
  }, [id])

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [logs])

  if (!status) {
    return (
      <div className="page-body" style={{ display: 'flex', alignItems: 'center', gap: 10, color: 'var(--color-text-muted)', paddingTop: 48 }}>
        <span className="spinner spinner-lg" /> Loading simulation...
      </div>
    )
  }

  const isCompleted = status.current_stage === 'completed'
  const isFailed = status.current_stage === 'failed'

  return (
    <div>
      <div className="page-header">
        <div>
          <h2 className="page-title" style={{ fontSize: 16, fontFamily: 'JetBrains Mono, monospace' }}>
            {status.simulation_id}
          </h2>
          <p className="page-subtitle">
            Region: <strong>{status.region_name}</strong> &nbsp;·&nbsp;
            Started: {status.started_at ? new Date(status.started_at).toLocaleString() : 'Pending'}
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          {isCompleted && (
            <button
              className="btn btn-accent btn-lg"
              onClick={() => navigate(`/simulations/${id}/visualize`)}
            >
              🌐 Open 3D Visualizer
            </button>
          )}
          {(isCompleted || isFailed) && (
            <button className="btn btn-ghost" onClick={() => navigate('/simulations/new')}>
              New Simulation
            </button>
          )}
        </div>
      </div>
      <div className="page-body">

        {/* Status banner */}
        {isCompleted && (
          <div className="alert alert-success" style={{ marginBottom: 16 }}>
            ✓ Simulation complete — all pipeline stages finished successfully.
          </div>
        )}
        {isFailed && (
          <div className="alert alert-error" style={{ marginBottom: 16 }}>
            ✗ Simulation failed: {status.error || 'See pipeline logs for details.'}
          </div>
        )}
        {!isCompleted && !isFailed && (
          <div className="alert alert-info" style={{ marginBottom: 16, display: 'flex', alignItems: 'center', gap: 10 }}>
            <span className="spinner" />
            Pipeline running — stage: <strong>{status.current_stage}</strong>
          </div>
        )}

        {/* Tab bar */}
        <div className="step-tabs" style={{ marginBottom: 20 }}>
          {(['progress','logs','results'] as const).map(tab => (
            <div
              key={tab}
              className={`step-tab${activeTab === tab ? ' active' : ''}`}
              onClick={() => setActiveTab(tab)}
              style={{ cursor: 'pointer' }}
            >
              {tab === 'progress' ? '◉ Progress' : tab === 'logs' ? '📋 Logs' : '📊 Results'}
            </div>
          ))}
        </div>

        {activeTab === 'progress' && <StatusSection status={status} />}

        {activeTab === 'logs' && (
          <div className="card">
            <div className="card-title">Pipeline Logs ({logs.length} lines)</div>
            <div className="log-viewer" ref={logRef}>
              {logs.map((line, i) => {
                const cls = line.includes('[ERROR]') || line.includes('Traceback') ? 'log-line-error'
                  : line.includes('✓') || line.includes('Done') || line.includes('complete') ? 'log-line-ok success'
                  : line.includes('WARNING') || line.includes('WARN') ? 'log-line-warn'
                  : 'log-line-ok'
                return <div key={i} className={cls}>{line || ' '}</div>
              })}
              {!isCompleted && !isFailed && (
                <div style={{ color: 'var(--color-primary)', marginTop: 4 }}>
                  <span className="pulse-dot" /> Processing...
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'results' && (
          isCompleted && results
            ? <ResultsSection results={results} />
            : isFailed
            ? <div className="alert alert-error">Simulation failed — no results available.</div>
            : <div className="alert alert-info"><span className="spinner" /> Waiting for simulation to complete...</div>
        )}
      </div>
    </div>
  )
}
