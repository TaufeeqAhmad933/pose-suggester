/**
 * CameraView.jsx
 * ---------------
 * Handles image input: file upload or webcam capture.
 * Calls onImageReady(file, previewUrl) when an image is selected.
 */

import { useRef, useState } from 'react'

export default function CameraView({ onImageReady, disabled }) {
  const fileInputRef  = useRef(null)
  const videoRef      = useRef(null)
  const [showCam, setShowCam]   = useState(false)
  const [stream, setStream]     = useState(null)
  const [dragOver, setDragOver] = useState(false)

  // ── File upload ────────────────────────────────────────────────────────────
  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    const url = URL.createObjectURL(file)
    onImageReady(file, url)
  }

  // ── Drag & drop ────────────────────────────────────────────────────────────
  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file && file.type.startsWith('image/')) {
      const url = URL.createObjectURL(file)
      onImageReady(file, url)
    }
  }

  // ── Webcam ─────────────────────────────────────────────────────────────────
  const startCamera = async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 1280, height: 720 }
      })
      setStream(s)
      setShowCam(true)
      // Attach stream to video after render
      setTimeout(() => {
        if (videoRef.current) videoRef.current.srcObject = s
      }, 100)
    } catch (err) {
      alert('Camera access denied or unavailable.')
    }
  }

  const stopCamera = () => {
    stream?.getTracks().forEach(t => t.stop())
    setStream(null)
    setShowCam(false)
  }

  const capturePhoto = () => {
    const video = videoRef.current
    if (!video) return
    const canvas = document.createElement('canvas')
    canvas.width  = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)
    canvas.toBlob(blob => {
      if (!blob) return
      const file = new File([blob], 'webcam_capture.jpg', { type: 'image/jpeg' })
      const url  = URL.createObjectURL(blob)
      onImageReady(file, url)
      stopCamera()
    }, 'image/jpeg', 0.92)
  }

  if (showCam) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          style={{ width: '100%', borderRadius: '16px', background: '#000' }}
        />
        <div style={{ display: 'flex', gap: '10px' }}>
          <button onClick={capturePhoto} style={btnStyle('#c8ff57', '#000')}>
            📸 Capture
          </button>
          <button onClick={stopCamera} style={btnStyle('var(--surface2)', 'var(--text)')}>
            Cancel
          </button>
        </div>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      {/* Drop zone */}
      <div
        onDragOver={e => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !disabled && fileInputRef.current?.click()}
        style={{
          border: `2px dashed ${dragOver ? 'var(--accent)' : 'var(--border)'}`,
          borderRadius: '20px',
          padding: '48px 24px',
          textAlign: 'center',
          cursor: disabled ? 'not-allowed' : 'pointer',
          transition: 'border-color 0.2s, background 0.2s',
          background: dragOver ? 'rgba(200,255,87,0.05)' : 'var(--surface)',
          opacity: disabled ? 0.5 : 1,
        }}
      >
        <div style={{ fontSize: '40px', marginBottom: '12px' }}>🖼️</div>
        <div style={{ fontWeight: 700, fontSize: '16px', marginBottom: '6px' }}>
          Drop your photo here
        </div>
        <div style={{ color: 'var(--muted)', fontSize: '13px', fontFamily: 'var(--font-mono)' }}>
          or click to browse · JPEG, PNG, WebP
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={handleFileChange}
          style={{ display: 'none' }}
          disabled={disabled}
        />
      </div>

      {/* Webcam button */}
      <button
        onClick={startCamera}
        disabled={disabled}
        style={btnStyle('var(--surface2)', 'var(--text)')}
      >
        📷 Use webcam instead
      </button>
    </div>
  )
}

function btnStyle(bg, color) {
  return {
    background: bg, color, border: '1px solid var(--border)',
    borderRadius: '12px', padding: '12px 20px',
    fontFamily: 'var(--font-display)', fontWeight: 600, fontSize: '14px',
    cursor: 'pointer', transition: 'opacity 0.2s',
    width: '100%',
  }
}