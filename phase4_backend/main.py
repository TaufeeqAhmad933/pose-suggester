"""
Phase 4 Backend — FastAPI Main
---------------------------------
Endpoints:
  POST /analyse      → upload image → scene + keypoints + pose suggestions
  GET  /health       → liveness check
  GET  /scenes       → list available scene labels

Run:
    uvicorn main:app --reload --port 8000
"""

import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.pose_estimator import get_pose_estimator
from services.scene_classifier import get_scene_classifier
from services.pose_suggester import get_pose_suggester

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PoseSuggester API",
    description="Upload a photo → get scene classification + skeleton pose suggestions",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Allowed image types
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_MB = 10


# ── Response models ───────────────────────────────────────────────────────────

class KeypointPoint(BaseModel):
    name: str
    x: float
    y: float
    visibility: float


class PoseSuggestion(BaseModel):
    rank: int
    score: float
    scene: str
    source_image: str
    keypoints: list[KeypointPoint]


class ScenePrediction(BaseModel):
    label: str
    category: str
    subcategory: str
    confidence: float


class AnalyseResponse(BaseModel):
    person_detected: bool
    scene: ScenePrediction | None
    person_keypoints: list[KeypointPoint] | None
    suggestions: list[PoseSuggestion]
    message: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "PoseSuggester API"}


@app.get("/scenes")
def list_scenes():
    library_dir = Path("pose_library")
    if not library_dir.exists():
        return {"scenes": [], "note": "Run phase3 build_pose_library.py first"}
    scenes = sorted({
        f.stem for f in library_dir.glob("*.joblib")
    })
    return {"scenes": scenes}


@app.post("/analyse", response_model=AnalyseResponse)
async def analyse(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Use JPEG, PNG, or WebP."
        )

    image_bytes = await file.read()

    # Validate file size
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB."
        )

    # ── 1. Detect person pose ─────────────────────────────────────────────────
    estimator = get_pose_estimator()
    person_kp = estimator.estimate(image_bytes)

    if person_kp is None:
        return AnalyseResponse(
            person_detected=False,
            scene=None,
            person_keypoints=None,
            suggestions=[],
            message="No person detected in the image. Please upload a photo with a person.",
        )

    person_kp_list = [
        KeypointPoint(name=name, **{k: v for k, v in vals.items() if k != "z"})
        for name, vals in person_kp.items()
        if "x" in vals and "y" in vals
    ]

    # ── 2. Classify scene ─────────────────────────────────────────────────────
    try:
        classifier = get_scene_classifier()
        scene_result = classifier.predict(image_bytes)
        scene = ScenePrediction(
            label=scene_result["label"],
            category=scene_result["category"],
            subcategory=scene_result["subcategory"],
            confidence=scene_result["confidence"],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # ── 3. Get pose suggestions ───────────────────────────────────────────────
    try:
        suggester = get_pose_suggester()
        raw_suggestions = suggester.suggest(
            scene_label=scene_result["label"],
            person_keypoints=person_kp,
            n_suggestions=3,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    suggestions = [
        PoseSuggestion(
            rank=s["rank"],
            score=s["score"],
            scene=s["scene"],
            source_image=s["source_image"],
            keypoints=[KeypointPoint(**kp) for kp in s["keypoints"]],
        )
        for s in raw_suggestions
    ]

    return AnalyseResponse(
        person_detected=True,
        scene=scene,
        person_keypoints=person_kp_list,
        suggestions=suggestions,
        message=f"Found {len(suggestions)} pose suggestions for {scene.category} / {scene.subcategory}",
    )