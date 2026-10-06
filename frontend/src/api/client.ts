// src/api/client.ts
import axios from 'axios'

const client = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
})

export default client

// ── Types ──────────────────────────────────────────────────────────────────

export type Region = {
  id: string
  name: string
  description: string
  geojson_available: boolean
  dem_available: boolean
  geojson_size_bytes: number | null
  dem_size_bytes: number | null
  reference_point: [number, number] | null
  resolution_m: number | null
  bounds: { minx: number; miny: number; maxx: number; maxy: number } | null
  thumbnail: string | null
}

export type PipelineStage = {
  name: string
  label: string
  status: 'waiting' | 'running' | 'done' | 'failed' | 'skipped'
  started_at: string | null
  completed_at: string | null
  error: string | null
}

export type SimulationStatus = {
  simulation_id: string
  region_id: string
  region_name: string
  created_at: string
  started_at: string | null
  completed_at: string | null
  current_stage: string
  stages: PipelineStage[]
  error: string | null
  run_dir: string | null
}

export type SimulationResults = {
  simulation_id: string
  region_id: string
  num_radars?: number
  radar_types?: Record<string, number>
  num_boundary_points?: number
  coverage_pct_any?: number
  uncoverable_pts?: number
  phase?: number
  solver_status?: string
  total_on_slots?: number
  total_void?: number
  void_pct?: number
  total_overlap?: number
  overlap_density?: number
  frequency_bands_used?: string[]
  baselines?: Record<string, any>
  output_files?: Record<string, string>
}

export type ValidationResult = {
  valid: boolean
  errors: string[]
  summary?: {
    num_types: number
    total_radars: number
    types: string[]
    frequency_bands: string[]
    per_type: Record<string, any>
  }
}

export type VisualizationInfo = {
  simulation_id: string
  czml_url: string
  border_url: string
  radar_meta_url: string
  schedule_url: string
  type_config_url: string
}

// ── API functions ──────────────────────────────────────────────────────────

export const api = {
  regions: {
    list: (): Promise<{ regions: Region[] }> => client.get('/regions/').then(r => r.data),
    get: (id: string): Promise<Region> => client.get(`/regions/${id}`).then(r => r.data),
    files: (id: string): Promise<{ geojson_available: boolean; dem_available: boolean; reference_point: any }> =>
      client.get(`/regions/${id}/files`).then(r => r.data),
  },

  simulations: {
    list: (): Promise<{ simulations: SimulationStatus[] }> =>
      client.get('/simulations/').then(r => r.data),

    create: (region_id: string, radar_config: object): Promise<{ simulation_id: string }> =>
      client.post('/simulations/', { region_id, radar_config }).then(r => r.data),

    status: (id: string): Promise<SimulationStatus> =>
      client.get(`/simulations/${id}/status`).then(r => r.data),

    logs: (id: string, last?: number): Promise<{ logs: string[] }> =>
      client.get(`/simulations/${id}/logs`, { params: { last: last ?? 500 } }).then(r => r.data),

    results: (id: string): Promise<SimulationResults> =>
      client.get(`/simulations/${id}/results`).then(r => r.data),

    visualization: (id: string): Promise<VisualizationInfo> =>
      client.get(`/simulations/${id}/visualization`).then(r => r.data),

    validateConfig: (config: object): Promise<ValidationResult> =>
      client.post('/simulations/validate-config', { config }).then(r => r.data),
  },
}
