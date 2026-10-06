// src/pages/Visualizer.tsx
// Loads the Cesium 3D visualizer for a completed simulation.
// Strategy: serve the existing index.html through the backend with CZML/GeoJSON
// URLs injected via query parameters. The iframe approach ensures the CesiumJS
// viewer (which uses complex global state and CDN imports) runs exactly as designed.

import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api, VisualizationInfo } from '../api/client'

export default function Visualizer() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [info, setInfo] = useState<VisualizationInfo | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!id) return
    api.simulations.visualization(id)
      .then(setInfo)
      .catch(e => {
        const msg = e?.response?.data?.detail || 'Failed to load visualization data.'
        setError(typeof msg === 'string' ? msg : JSON.stringify(msg))
      })
      .finally(() => setLoading(false))
  }, [id])

  if (loading) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100vh', background: 'var(--color-bg)', color: 'var(--color-text-muted)',
        gap: 14, flexDirection: 'column',
      }}>
        <span className="spinner spinner-lg" />
        <span>Loading visualization...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100vh', background: 'var(--color-bg)', padding: 32,
        flexDirection: 'column', gap: 16,
      }}>
        <div className="alert alert-error" style={{ maxWidth: 520 }}>
          <strong>⚠ Visualization Error</strong><br />{error}
        </div>
        <button className="btn btn-ghost" onClick={() => navigate(`/simulations/${id}`)}>
          ← Back to Simulation
        </button>
      </div>
    )
  }

  if (!info) return null

  // Build the viewer URL — we serve a modified viewer HTML that accepts URL params
  const czmlParam = encodeURIComponent(info.czml_url)
  const borderParam = encodeURIComponent(info.border_url)
  const viewerUrl = `/viewer.html?czml=${czmlParam}&geojson=${borderParam}&sim=${id}`

  return (
    <div style={{ position: 'fixed', inset: 0, background: '#000', zIndex: 0 }}>
      {/* Back button overlay */}
      <div style={{
        position: 'absolute', top: 14, left: 14, zIndex: 1000,
        display: 'flex', gap: 8, alignItems: 'center',
      }}>
        <button
          className="btn btn-ghost"
          style={{
            background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(8px)',
            borderColor: 'rgba(255,255,255,0.15)',
          }}
          onClick={() => navigate(`/simulations/${id}`)}
        >
          ← Back to Results
        </button>
        <div style={{
          background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(8px)',
          border: '1px solid rgba(255,255,255,0.12)',
          borderRadius: 8, padding: '7px 14px',
          fontSize: 11, color: '#94a3b8', fontFamily: 'JetBrains Mono, monospace',
        }}>
          {id}
        </div>
      </div>

      <iframe
        src={viewerUrl}
        style={{ width: '100%', height: '100%', border: 'none' }}
        title="3D Radar Coverage Visualizer"
        allow="fullscreen"
      />
    </div>
  )
}
