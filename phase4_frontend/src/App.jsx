/**
 * App.jsx — PoseSuggester Main Application
 * ------------------------------------------
 * Aesthetic: Dark editorial / brutalist-refined.
 * Typography: Syne (display) + DM Mono (data/labels).
 * Palette: Near-black bg · acid green accent · electric blue · coral.
 */

import { useState } from 'react'
import CameraView      from './components/CameraView.jsx'
import SkeletonOverlay from './components/SkeletonOverlay.jsx'
import PoseCard        from './components/PoseCard.jsx'
import SceneLabel      from './components/SceneLabel.jsx'
import { usePoseSuggestions } from './hooks/usePoseSuggestions.js'

export default function App() {
  const [imageFile, setImageFile]       = useState(null)
  const [previewUrl, setPreviewUrl]     = useState(null)
  const [activePoseIdx, setActivePoseIdx] = useState(0)

  const {
    loading, error, scene,
    personKeypoints, suggestions,
    personDetected, message,
    analyse, reset,
  } = usePoseSuggestions()

  const handleImageReady = (file, url) => {
    setImageFile(file)
    setPreviewUrl(url)
    setActivePoseIdx(0)
    reset()
  }

  const handleAnalyse = () => {
    if (imageFile) analyse(imageFile)
  }

  const handleReset = () => {
    setImageFile(null)
    setPreviewUrl(null)
    setActivePoseIdx(0)
    reset()
  }

  const activeSuggestion = suggestions[activePoseIdx] || null
  const hasResults = suggestions.length > 0

  return (
    <div style={{
      minHeight: '100vh',
      background: 'var(--bg)',
      display: 'flex', flexDirection: 'column',
    }}>
      {/* ── Header ── */}
      <header style={{
        padding: '24px 40px',
        borderBottom: '1px solid var(--border)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '12px' }}>
          <span style={{
            fontFamily: 'var(--font-display)', fontWeight: 800,
            fontSize: '22px', letterSpacing: '-0.03em',
          }}>
            POSE<span style={{ color: 'var(--accent)' }}>·</span>SUGGESTER
          </span>
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: '11px',
            color: 'var(--muted)', letterSpacing: '0.08em',
          }}>
            v1.0
          </span>
        </div>
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: '12px',
          color: 'var(--muted)',
        }}>
          AI-powered pose guidance
        </div>
      </header>

      {/* ── Main layout ── */}
      <main style={{
        flex: 1, display: 'grid',
        gridTemplateColumns: previewUrl ? '1fr 380px' : '1fr',
        gap: '0',
        maxWidth: '1400px', margin: '0 auto', width: '100%',
        padding: '40px',
      }}>

        {/* ── Left: image area ── */}
        <div style={{ paddingRight: previewUrl ? '40px' : '0' }}>

          {!previewUrl ? (
            /* Upload state */
            <div style={{ maxWidth: '560px', margin: '0 auto' }}>
              <div style={{ marginBottom: '40px' }}>
                <h1 style={{
                  fontWeight: 800, fontSize: 'clamp(32px, 5vw, 52px)',
                  lineHeight: 1.05, letterSpacing: '-0.04em',
                  marginBottom: '16px',
                }}>
                  Strike the<br />
                  <span style={{ color: 'var(--accent)' }}>perfect pose</span>
                </h1>
                <p style={{
                  color: 'var(--muted)', fontSize: '15px', lineHeight: 1.6,
                  fontFamily: 'var(--font-mono)', fontWeight: 300,
                }}>
                  Upload a photo — we'll detect your scene, analyse your position,
                  and suggest the 3 best poses for that background.
                </p>
              </div>
              <CameraView onImageReady={handleImageReady} disabled={loading} />
            </div>
          ) : (
            /* Image + overlay */
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

              {/* Scene label */}
              {scene && <SceneLabel scene={scene} />}

              {/* Canvas with skeleton overlay */}
              <SkeletonOverlay
                imageSrc={previewUrl}
                personKeypoints={personKeypoints}
                activeSuggestion={activeSuggestion}
              />

              {/* No person warning */}
              {personDetected === false && (
                <div style={{
                  background: 'rgba(255,107,107,0.1)',
                  border: '1px solid var(--accent3)',
                  borderRadius: '12px', padding: '16px 20px',
                  color: 'var(--accent3)', fontFamily: 'var(--font-mono)', fontSize: '13px',
                }}>
                  ⚠️ {message}
                </div>
              )}

              {/* Error */}
              {error && (
                <div style={{
                  background: 'rgba(255,107,107,0.1)',
                  border: '1px solid var(--accent3)',
                  borderRadius: '12px', padding: '16px 20px',
                  color: 'var(--accent3)', fontFamily: 'var(--font-mono)', fontSize: '13px',
                }}>
                  ❌ {error}
                </div>
              )}

              {/* Action buttons */}
              <div style={{ display: 'flex', gap: '12px' }}>
                {!hasResults && !loading && (
                  <button
                    onClick={handleAnalyse}
                    disabled={loading}
                    style={{
                      flex: 1, background: 'var(--accent)', color: '#000',
                      border: 'none', borderRadius: '12px',
                      padding: '16px', fontFamily: 'var(--font-display)',
                      fontWeight: 700, fontSize: '15px', cursor: 'pointer',
                      transition: 'opacity 0.2s',
                    }}
                  >
                    ✦ Suggest Poses
                  </button>
                )}
                <button
                  onClick={handleReset}
                  style={{
                    flex: hasResults ? 0 : 1,
                    background: 'var(--surface2)', color: 'var(--text)',
                    border: '1px solid var(--border)', borderRadius: '12px',
                    padding: '16px 24px', fontFamily: 'var(--font-display)',
                    fontWeight: 600, fontSize: '15px', cursor: 'pointer',
                  }}
                >
                  ↺ New Photo
                </button>
              </div>
            </div>
          )}
        </div>

        {/* ── Right: pose suggestion panel ── */}
        {previewUrl && (
          <div style={{
            borderLeft: '1px solid var(--border)',
            paddingLeft: '40px',
            display: 'flex', flexDirection: 'column', gap: '20px',
          }}>
            <div>
              <div style={{
                fontFamily: 'var(--font-mono)', fontSize: '11px',
                color: 'var(--muted)', letterSpacing: '0.1em',
                marginBottom: '8px',
              }}>
                POSE SUGGESTIONS
              </div>
              <div style={{ fontWeight: 700, fontSize: '20px' }}>
                {loading ? 'Analysing…' : hasResults ? `${suggestions.length} poses found` : 'Upload & analyse'}
              </div>
            </div>

            {/* Loading spinner */}
            {loading && (
              <div style={{
                display: 'flex', flexDirection: 'column', gap: '16px',
                alignItems: 'center', padding: '40px 0',
                color: 'var(--muted)', fontFamily: 'var(--font-mono)', fontSize: '13px',
              }}>
                <div style={{
                  width: '40px', height: '40px',
                  border: '3px solid var(--border)',
                  borderTop: '3px solid var(--accent)',
                  borderRadius: '50%',
                  animation: 'spin 0.8s linear infinite',
                }} />
                Detecting pose & scene…
                <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
              </div>
            )}

            {/* Pose cards */}
            {hasResults && suggestions.map((s, i) => (
              <PoseCard
                key={s.rank}
                suggestion={s}
                isActive={i === activePoseIdx}
                onClick={() => setActivePoseIdx(i)}
              />
            ))}

            {/* Instructions if not yet analysed */}
            {!loading && !hasResults && !error && personDetected === null && (
              <div style={{
                color: 'var(--muted)', fontFamily: 'var(--font-mono)',
                fontSize: '13px', lineHeight: 1.7,
                border: '1px dashed var(--border)',
                borderRadius: '12px', padding: '20px',
              }}>
                Hit <strong style={{ color: 'var(--accent)' }}>✦ Suggest Poses</strong> to:<br /><br />
                1. Detect your body position<br />
                2. Classify the background scene<br />
                3. Get 3 tailored pose overlays
              </div>
            )}

            {/* Colour legend */}
            {hasResults && (
              <div style={{
                borderTop: '1px solid var(--border)', paddingTop: '16px',
                display: 'flex', flexDirection: 'column', gap: '8px',
              }}>
                <div style={{
                  fontFamily: 'var(--font-mono)', fontSize: '11px',
                  color: 'var(--muted)', letterSpacing: '0.08em', marginBottom: '4px',
                }}>
                  COLOUR KEY
                </div>
                {['#c8ff57','#57c8ff','#ff6b6b'].slice(0, suggestions.length).map((c, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <div style={{ width: '32px', height: '3px', background: c, borderRadius: '2px' }} />
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--muted)' }}>
                      Pose {i + 1}
                    </span>
                  </div>
                ))}
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <div style={{ width: '32px', height: '3px', background: 'rgba(255,255,255,0.35)', borderRadius: '2px' }} />
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--muted)' }}>
                    Your current pose
                  </span>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}