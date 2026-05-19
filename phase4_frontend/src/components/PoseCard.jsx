/**
 * PoseCard.jsx
 * Shows a single pose suggestion card with its mini skeleton canvas.
 */

import { useEffect, useRef } from 'react'
import { drawSkeleton } from '../utils/drawSkeleton.js'

const COLORS = ['#c8ff57', '#57c8ff', '#ff6b6b']

export default function PoseCard({ suggestion, isActive, onClick }) {
  const canvasRef = useRef(null)
  const color     = COLORS[(suggestion.rank - 1) % COLORS.length]

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !suggestion.keypoints) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    drawSkeleton(ctx, suggestion.keypoints, canvas.width, canvas.height, {
      boneColor:   color,
      jointColor:  '#ffffff',
      boneWidth:   2.5,
      jointRadius: 4,
      alpha:       1,
    })
  }, [suggestion, color])

  return (
    <div
      onClick={onClick}
      style={{
        cursor: 'pointer',
        border: `2px solid ${isActive ? color : 'var(--border)'}`,
        borderRadius: '16px',
        background: isActive ? `${color}10` : 'var(--surface)',
        padding: '14px',
        transition: 'all 0.2s ease',
        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px',
      }}
    >
      {/* Rank badge */}
      <div style={{
        width: '100%', display: 'flex',
        justifyContent: 'space-between', alignItems: 'center',
      }}>
        <span style={{
          fontFamily: 'var(--font-mono)', fontSize: '11px',
          color: color, letterSpacing: '0.08em', fontWeight: 500,
        }}>
          POSE {suggestion.rank}
        </span>
        <span style={{
          fontFamily: 'var(--font-mono)', fontSize: '11px',
          color: 'var(--muted)',
        }}>
          {Math.round(suggestion.score * 100)}% match
        </span>
      </div>

      {/* Mini skeleton canvas */}
      <canvas
        ref={canvasRef}
        width={140}
        height={180}
        style={{
          background: 'var(--surface2)',
          borderRadius: '10px',
          width: '100%', height: '150px',
        }}
      />

      {isActive && (
        <div style={{
          width: '100%', textAlign: 'center',
          fontFamily: 'var(--font-mono)', fontSize: '10px',
          color: color, letterSpacing: '0.05em',
        }}>
          ● SHOWING ON IMAGE
        </div>
      )}
    </div>
  )
}