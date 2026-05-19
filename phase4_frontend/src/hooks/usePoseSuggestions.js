/**
 * usePoseSuggestions.js
 * -----------------------
 * Custom hook that POSTs an image file to the /analyse endpoint
 * and returns scene classification + pose suggestions.
 */

import { useState, useCallback } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || ''

export function usePoseSuggestions() {
  const [state, setState] = useState({
    loading:           false,
    error:             null,
    scene:             null,   // { label, category, subcategory, confidence }
    personKeypoints:   null,   // array of {name, x, y, visibility}
    suggestions:       [],     // array of pose suggestion objects
    personDetected:    null,
    message:           '',
  })

  const analyse = useCallback(async (imageFile) => {
    setState(s => ({ ...s, loading: true, error: null }))

    const formData = new FormData()
    formData.append('file', imageFile)

    try {
      const res = await fetch(`${API_BASE}/analyse`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail || `HTTP ${res.status}`)
      }

      const data = await res.json()
      setState({
        loading:         false,
        error:           null,
        scene:           data.scene,
        personKeypoints: data.person_keypoints,
        suggestions:     data.suggestions,
        personDetected:  data.person_detected,
        message:         data.message,
      })
    } catch (err) {
      setState(s => ({
        ...s,
        loading: false,
        error:   err.message || 'Something went wrong.',
      }))
    }
  }, [])

  const reset = useCallback(() => {
    setState({
      loading: false, error: null, scene: null,
      personKeypoints: null, suggestions: [], personDetected: null, message: '',
    })
  }, [])

  return { ...state, analyse, reset }
}