/**
 * SkeletonOverlay.jsx
 * --------------------
 * Renders the uploaded image with skeleton overlays on a canvas:
 *   - Grey/dim skeleton = detected person's current pose
 *   - Coloured skeleton = selected suggested pose
 */

import { useEffect, useRef } from 'react'
import { drawPersonSkeleton, drawSkeleton } from '../utils/drawSkeleton.js'

const COLORS = ['#c8ff57', '#57c8ff', '#ff6b6b']

export default function SkeletonOverlay({
  imageSrc,
  personKeypoints,
  activeSuggestion,
}) {
  const canvasRef = useRef(null)
  const imgRef    = useRef(null)

  // Draw whenever image or keypoints change
  useEffect(() => {
    const canvas = canvasRef.current
    const img    = imgRef.current
    if (!canvas || !img || !img.complete) return

    const W = img.naturalWidth
    const H = img.naturalHeight
    canvas.width  = W
    canvas.height = H

    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, W, H)
    ctx.drawImage(img, 0, 0, W, H)

    // Person's current pose (faint)
    if (personKeypoints) {
      drawPersonSkeleton(ctx, personKeypoints, W, H)
    }

    // Active suggestion overlay (vivid)
    if (activeSuggestion) {
      const color = COLORS[(activeSuggestion.rank - 1) % COLORS.length]
      drawSkeleton(ctx, activeSuggestion.keypoints, W, H, {
        boneColor:   color,
        jointColor:  '#ffffff',
        boneWidth:   4,
        jointRadius: 6,
        alpha:       0.92,
      })
    }
  }, [imageSrc, personKeypoints, activeSuggestion])

  const handleImageLoad = () => {
    // Re-trigger draw after image loads
    const canvas = canvasRef.current
    const img    = imgRef.current
    if (!canvas || !img) return
    const W = img.naturalWidth
    const H = img.naturalHeight
    canvas.width  = W
    canvas.height = H
    const ctx = canvas.getContext('2d')
    ctx.drawImage(img, 0, 0, W, H)
    if (personKeypoints) drawPersonSkeleton(ctx, personKeypoints, W, H)
    if (activeSuggestion) {
      const color = COLORS[(activeSuggestion.rank - 1) % COLORS.length]
      drawSkeleton(ctx, activeSuggestion.keypoints, W, H, {
        boneColor: color, jointColor: '#ffffff',
        boneWidth: 4, jointRadius: 6, alpha: 0.92,
      })
    }
  }

  return (
    <div style={{ position: 'relative', width: '100%' }}>
      {/* Hidden img for loading the source */}
      <img
        ref={imgRef}
        src={imageSrc}
        onLoad={handleImageLoad}
        style={{ display: 'none' }}
        alt=""
      />
      {/* Visible canvas */}
      <canvas
        ref={canvasRef}
        style={{
          width: '100%', height: 'auto',
          borderRadius: '16px',
          display: 'block',
        }}
      />
      {/* Legend */}
      {activeSuggestion && (
        <div style={{
          position: 'absolute', bottom: '12px', left: '12px',
          background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(8px)',
          borderRadius: '8px', padding: '8px 12px',
          display: 'flex', flexDirection: 'column', gap: '5px',
        }}>
          <LegendItem color="rgba(255,255,255,0.4)" label="Your current pose" />
          <LegendItem
            color={COLORS[(activeSuggestion.rank - 1) % COLORS.length]}
            label={`Suggested pose ${activeSuggestion.rank}`}
          />
        </div>
      )}
    </div>
  )
}

function LegendItem({ color, label }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <div style={{
        width: '24px', height: '3px', background: color, borderRadius: '2px',
      }} />
      <span style={{
        fontFamily: 'var(--font-mono)', fontSize: '11px', color: '#e0e0e0',
      }}>
        {label}
      </span>
    </div>
  )
}