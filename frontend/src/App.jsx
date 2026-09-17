import React, { useEffect, useState } from 'react'

export default function App() {
  const [siteName, setSiteName] = useState('')
  const [cameraId, setCameraId] = useState('')
  const [videoFile, setVideoFile] = useState(null)
  const [analysisId, setAnalysisId] = useState(null)
  const [status, setStatus] = useState(null)
  const [progress, setProgress] = useState(0)
  const [results, setResults] = useState(null)

  // Submit analysis
  const submit = async (e) => {
    e.preventDefault()
    if (!videoFile || !siteName || !cameraId) return
    const form = new FormData()
    form.append('video', videoFile)
    form.append('site_name', siteName)
    form.append('camera_id', cameraId)
    try {
      const res = await fetch('/api/v1/analysis', {
        method: 'POST',
        body: form,
      })
      const data = await res.json()
      if (data.analysis_id) {
        setAnalysisId(data.analysis_id)
        setStatus(data.status)
      }
    } catch (err) {
      console.error('Upload failed', err)
    }
  }

  // Polling for status if we have an analysis id
  useEffect(() => {
    let timer = null
    if (analysisId) {
      timer = setInterval(async () => {
        try {
          const r = await fetch(`/api/v1/analysis/${analysisId}`)
          if (r.ok) {
            const j = await r.json()
            setStatus(j.status)
            setProgress(j.progress || 0)
            if (j.results) setResults(j.results)
            if (j.status === 'completed' || j.status === 'failed') {
              clearInterval(timer)
            }
          }
        } catch (e) {
          // ignore
        }
      }, 1000)
    }
    return () => clearInterval(timer)
  }, [analysisId])

  const reportLink = analysisId ? `/api/v1/analysis/${analysisId}/report` : null

  return (
    <div className="container">
      <header className="header">
        <h1>RakshAI Frontend</h1>
        <p>Submit a video for PPE safety analysis. Backend: FastAPI with ONNX model.</p>
      </header>

      <section className="card">
        <h2>Upload & Start Analysis</h2>
        <form onSubmit={submit} className="form">
          <div className="field">
            <label>Site Name</label>
            <input type="text" value={siteName} onChange={(e) => setSiteName(e.target.value)} placeholder="Site A" />
          </div>
          <div className="field">
            <label>Camera ID</label>
            <input type="text" value={cameraId} onChange={(e) => setCameraId(e.target.value)} placeholder="CAM-01" />
          </div>
          <div className="field">
            <label>Video File</label>
            <input type="file" accept="video/*" onChange={(e) => setVideoFile(e.target.files?.[0] ?? null)} />
          </div>
          <button className="button" type="submit" disabled={!videoFile || !siteName || !cameraId}>Start Analysis</button>
        </form>
      </section>

      {analysisId && (
        <section className="card">
          <h2>Analysis Status</h2>
          <p>Analysis ID: <code>{analysisId}</code></p>
          <p>Status: {status ?? 'pending'}</p>
          <div className="progress-bar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}>
            <div className="fill" style={{ width: `${progress || 0}%` }} />
          </div>

          {results && (
            <div className="results">
              <h3>Results</h3>
              <p>Duration: {results.duration_seconds} s</p>
              <p>Frames: {results.frames_processed}</p>
              <div className="detections">
                <h4>Detections</h4>
                {((results.detections) || []).length === 0 ? (
                  <p>No detections yet or model unavailable.</p>
                ) : (
                  results.detections.map((frameGroup, idx) => (
                    <div key={idx} className="frame-group">
                      <strong>Frame {frameGroup.frame}</strong>
                      <ul>
                        {frameGroup.detections.map((d, j) => (
                          <li key={j}>
                            {d.class_name} - {Math.round((d.confidence || 0) * 100) / 100}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))
                )}
              </div>
              {reportLink && (
                <a href={reportLink} target="_blank" rel="noreferrer" className="button link">Download PDF Report</a>
              )}
            </div>
          )}
        </section>
      )}
    </div>
  )
}
