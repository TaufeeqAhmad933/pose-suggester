# PoseSuggester 🧍‍♂️✦

> Upload a photo → detect your scene + body → get 3 tailored skeleton pose suggestions overlaid on your image.



## Project Structure

```
pose-suggester/
├── phase1_data/         Data collection, keypoint extraction, augmentation
├── phase2_classifier/   MobileNetV2 scene classifier (6 classes)
├── phase3_suggester/    KNN pose suggestion engine
├── phase4_backend/      FastAPI backend (ties phases 1-3 together)
└── phase4_frontend/     React web app (upload/webcam + skeleton overlay)
```



## Quickstart (in order)

### Phase 1 — Data & Annotations

```bash
cd phase1_data
pip install -r requirements.txt

# 1. Put 50 images in each folder:
#    dataset/cafe_indoor/full_body/
#    dataset/cafe_indoor/waist_up/
#    dataset/nature/full_body/
#    dataset/nature/waist_up/
#    dataset/urban_street/full_body/
#    dataset/urban_street/waist_up/

# 2. Extract MediaPipe keypoints → annotations/
python extract_keypoints.py

# 3. Augment dataset to ~300 per subcategory
python augment_dataset.py

# 4. Sanity check
python verify_dataset.py --visualise
```

### Phase 2 — Train Scene Classifier

```bash
cd phase2_classifier
pip install -r requirements.txt
python train.py                         # trains MobileNetV2, saves to models/
python evaluate.py                      # confusion matrix + accuracy
python predict.py --image my_photo.jpg  # test single image
```

### Phase 3 — Build Pose Library

```bash
cd phase3_suggester
pip install -r requirements.txt
python build_pose_library.py            # builds KNN index → pose_library/

# Copy models + pose_library into backend
cp -r pose_library/ ../phase4_backend/pose_library/
cp -r ../phase2_classifier/models/ ../phase4_backend/models/
```

### Phase 4 — Run the App

**Backend:**
```bash
cd phase4_backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd phase4_frontend
npm install
npm run dev                             # opens at http://localhost:5173
```

Open **http://localhost:5173** and start posing! 📸



## How It Works

```
User uploads photo
        ↓
FastAPI /analyse endpoint
        ↓
MediaPipe Pose  → person's current keypoints (17 joints)
        ↓
MobileNetV2     → scene label (e.g. "cafe_indoor / waist_up")
        ↓
KNN Pose Engine → top 3 similar-but-better poses from your dataset
        ↓
React Canvas    → draws skeleton overlays on the image
```



## Scene Categories

| Category | Subcategories |
|---|---|
| ☕ Cafe Indoor | Full Body, Waist Up |
| 🌿 Nature | Full Body, Waist Up |
| 🏙️ Urban Street | Full Body, Waist Up |



## API Endpoints

| Method | Route | Description |
|---|---|---|
| POST | `/analyse` | Upload image → scene + keypoints + suggestions |
| GET | `/scenes` | List available scene labels |
| GET | `/health` | Liveness check |



## Tech Stack

- **Pose Detection:** MediaPipe Pose (17 landmarks)
- **Scene Classifier:** MobileNetV2 (fine-tuned, TensorFlow/Keras)
- **Pose Engine:** scikit-learn NearestNeighbors (cosine distance)
- **Backend:** FastAPI + Uvicorn
- **Frontend:** React + Vite + HTML5 Canvas
- **Augmentation:** albumentations
