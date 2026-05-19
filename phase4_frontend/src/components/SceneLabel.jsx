/**
 * SceneLabel.jsx
 * Shows the detected scene category with confidence badge.
 */

const SCENE_ICONS = {
  cafe_indoor:  '☕',
  nature:       '🌿',
  urban_street: '🏙️',
}

const SUBCAT_LABELS = {
  full_body: 'Full Body',
  waist_up:  'Waist Up',
}

export default function SceneLabel({ scene }) {
  if (!scene) return null

  const icon = SCENE_ICONS[scene.category] || '📍'
  const catLabel = scene.category.replace('_', ' ')
  const subLabel = SUBCAT_LABELS[scene.subcategory] || scene.subcategory
  const conf = Math.round(scene.confidence * 100)

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: '12px',
      background: 'var(--surface2)', border: '1px solid var(--border)',
      borderRadius: '12px', padding: '12px 18px',
    }}>
      <span style={{ fontSize: '28px' }}>{icon}</span>
      <div>
        <div style={{ fontWeight: 700, fontSize: '15px', textTransform: 'capitalize' }}>
          {catLabel}
        </div>
        <div style={{ color: 'var(--muted)', fontSize: '12px', fontFamily: 'var(--font-mono)' }}>
          {subLabel} · {conf}% confidence
        </div>
      </div>
      <div style={{
        marginLeft: 'auto', background: 'var(--accent)', color: '#000',
        borderRadius: '6px', padding: '4px 10px',
        fontFamily: 'var(--font-mono)', fontSize: '12px', fontWeight: 500,
      }}>
        {conf}%
      </div>
    </div>
  )
}