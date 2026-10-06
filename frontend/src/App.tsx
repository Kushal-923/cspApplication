// src/App.tsx
import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import NewSimulation from './pages/NewSimulation'
import SimulationDetail from './pages/SimulationDetail'
import Visualizer from './pages/Visualizer'
import './index.css'

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <h1>🛡 Radar CSP</h1>
        <p>Border Radar Optimization</p>
      </div>
      <nav className="sidebar-nav">
        <NavLink
          to="/"
          end
          className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
        >
          <span>⊞</span> Dashboard
        </NavLink>
        <NavLink
          to="/simulations/new"
          className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
        >
          <span>＋</span> New Simulation
        </NavLink>

        <div className="nav-section-label">Algorithms</div>
        <div className="nav-item" style={{ cursor: 'default', opacity: 0.5 }}>
          <span>◉</span> Placement (CP-SAT)
        </div>
        <div className="nav-item" style={{ cursor: 'default', opacity: 0.5 }}>
          <span>◉</span> Coverage Matrix
        </div>
        <div className="nav-item" style={{ cursor: 'default', opacity: 0.5 }}>
          <span>◉</span> ILP Scheduler
        </div>

        <div className="nav-section-label">About</div>
        <a
          href="https://github.com/Kushal-923/cspProject"
          target="_blank"
          rel="noreferrer"
          className="nav-item"
        >
          <span>⇱</span> GitHub Repository
        </a>
      </nav>
    </aside>
  )
}

function AppShell({ children }: { children: React.ReactNode }) {
  const location = useLocation()
  const isVisualizer = location.pathname.includes('/visualize')

  if (isVisualizer) {
    return <>{children}</>
  }

  return (
    <div className="app-shell">
      <Sidebar />
      <main className="main-content">{children}</main>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AppShell>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/simulations/new" element={<NewSimulation />} />
          <Route path="/simulations/:id" element={<SimulationDetail />} />
          <Route path="/simulations/:id/visualize" element={<Visualizer />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  )
}
