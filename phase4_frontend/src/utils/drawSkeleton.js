/**
 * drawSkeleton.js
 * ----------------
 * Draws a stick-figure skeleton on an HTML5 Canvas given MediaPipe keypoints.
 *
 * Keypoints format (from API):
 *   [{ name: "nose", x: 0.51, y: 0.12, visibility: 0.99 }, ...]
 *
 * x, y are normalised [0,1] relative to image dimensions.
 */

// Pairs of [landmarkA, landmarkB] to draw as bones
export const SKELETON_CONNECTIONS = [
  // Torso
  ["left_shoulder",  "right_shoulder"],
  ["left_shoulder",  "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip",       "right_hip"],
  // Left arm
  ["left_shoulder",  "left_elbow"],
  ["left_elbow",     "left_wrist"],
  // Right arm
  ["right_shoulder", "right_elbow"],
  ["right_elbow",    "right_wrist"],
  // Left leg
  ["left_hip",       "left_knee"],
  ["left_knee",      "left_ankle"],
  // Right leg
  ["right_hip",      "right_knee"],
  ["right_knee",     "right_ankle"],
  // Head
  ["nose",           "left_eye"],
  ["nose",           "right_eye"],
  ["left_eye",       "left_ear"],
  ["right_eye",      "right_ear"],
  ["nose",           "left_shoulder"],
  ["nose",           "right_shoulder"],
]

const VIS_THRESHOLD = 0.35

/**
 * Convert keypoints array to a name→point map.
 */
function toMap(keypoints) {
  const map = {}
  keypoints.forEach(kp => { map[kp.name] = kp })
  return map
}

/**
 * Draw a single skeleton.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {Array} keypoints        - array of {name, x, y, visibility}
 * @param {number} imgW            - canvas / image width in px
 * @param {number} imgH            - canvas / image height in px
 * @param {object} style           - optional style overrides
 */
export function drawSkeleton(ctx, keypoints, imgW, imgH, style = {}) {
  const {
    boneColor   = '#c8ff57',
    jointColor  = '#ffffff',
    boneWidth   = 3,
    jointRadius = 5,
    alpha       = 0.9,
  } = style

  const kpMap = toMap(keypoints)

  ctx.save()
  ctx.globalAlpha = alpha

  // Draw bones
  ctx.strokeStyle = boneColor
  ctx.lineWidth   = boneWidth
  ctx.lineCap     = 'round'

  for (const [nameA, nameB] of SKELETON_CONNECTIONS) {
    const a = kpMap[nameA]
    const b = kpMap[nameB]
    if (!a || !b) continue
    if (a.visibility < VIS_THRESHOLD || b.visibility < VIS_THRESHOLD) continue

    ctx.beginPath()
    ctx.moveTo(a.x * imgW, a.y * imgH)
    ctx.lineTo(b.x * imgW, b.y * imgH)
    ctx.stroke()
  }

  // Draw joints
  ctx.fillStyle = jointColor
  for (const kp of keypoints) {
    if (kp.visibility < VIS_THRESHOLD) continue
    ctx.beginPath()
    ctx.arc(kp.x * imgW, kp.y * imgH, jointRadius, 0, Math.PI * 2)
    ctx.fill()
  }

  ctx.restore()
}

/**
 * Draw multiple skeleton suggestions, each in a different colour.
 */
const SUGGESTION_COLORS = ['#c8ff57', '#57c8ff', '#ff6b6b']

export function drawSuggestions(ctx, suggestions, imgW, imgH) {
  suggestions.forEach((suggestion, idx) => {
    drawSkeleton(ctx, suggestion.keypoints, imgW, imgH, {
      boneColor:   SUGGESTION_COLORS[idx % SUGGESTION_COLORS.length],
      jointColor:  '#ffffff',
      boneWidth:   3,
      jointRadius: 5,
      alpha:       0.85,
    })
  })
}

/**
 * Draw the detected person's current skeleton (dimmer, grey).
 */
export function drawPersonSkeleton(ctx, keypoints, imgW, imgH) {
  drawSkeleton(ctx, keypoints, imgW, imgH, {
    boneColor:   'rgba(255,255,255,0.35)',
    jointColor:  'rgba(255,255,255,0.5)',
    boneWidth:   2,
    jointRadius: 4,
    alpha:       1,
  })
}