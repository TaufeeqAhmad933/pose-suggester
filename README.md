
```markdown
# PoseSuggester

PoseSuggester is a full-stack computer vision pipeline that analyzes scene context and user posture to generate tailored, overlaid skeleton pose suggestions. By combining deep learning-based scene classification with spatial keypoint matching, the application guides users toward optimal poses for their specific environment.

## Table of Contents
- [Features & Architecture](#features--architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quickstart Guide](#quickstart-guide)
- [API Reference](#api-reference)
- [Results](#results)

## Features & Architecture

The system operates through a sequential processing pipeline:

1. **Input & Pose Detection:** User uploads a photo (or uses webcam). MediaPipe Pose extracts a 17-point joint landmark skeleton.
2. **Context Analysis:** A fine-tuned MobileNetV2 model classifies the scene into specific environmental categories (e.g., Cafe, Nature, Urban) and framing types (Full Body vs. Waist Up).
3. **Recommendation Engine:** A K-Nearest Neighbors (KNN) algorithm calculates cosine distance against a custom pose library to retrieve the top 3 optimal, context-aware poses.
4. **Rendering:** A React-based HTML5 Canvas draws the recommended skeleton overlays directly onto the user's image.

### Scene Categories
| Category | Supported Framing |
| :--- | :--- |
| ☕ **Cafe Indoor** | Full Body, Waist Up |
| 🌿 **Nature** | Full Body, Waist Up |
| 🏙️ **Urban Street** | Full Body, Waist Up |

## Tech Stack

*   **Machine Learning & CV:** TensorFlow/Keras (MobileNetV2), MediaPipe Pose, scikit-learn (KNN), Albumentations
*   **Backend:** FastAPI, Uvicorn, Python
*   **Frontend:** React, Vite, HTML5 Canvas

## Project Structure

```text
pose-suggester/
├── phase1_data/         # Data collection, keypoint extraction, and augmentation
├── phase2_classifier/   # MobileNetV2 scene classifier training and evaluation
├── phase3_suggester/    # KNN pose suggestion engine and library generation
├── phase4_backend/      # FastAPI backend handling inference and routing
└── phase4_frontend/     # React application for user interaction and overlay

```

## Quickstart Guide

Follow these steps in sequence to train the models, build the library, and run the application.

### Phase 1: Data Preparation

```bash
cd phase1_data
pip install -r requirements.txt

# 1. Populate the dataset/ directory with 50 images per subcategory:
#    dataset/[cafe_indoor|nature|urban_street]/[full_body|waist_up]/

# 2. Extract MediaPipe keypoints to the annotations/ folder
python extract_keypoints.py

# 3. Apply augmentations to expand the dataset to ~300 images per subcategory
python augment_dataset.py

# 4. Run a visual sanity check
python verify_dataset.py --visualise

```

### Phase 2: Train the Scene Classifier

```bash
cd phase2_classifier
pip install -r requirements.txt

# Train the MobileNetV2 model (saves to models/)
python train.py

# Evaluate accuracy and generate a confusion matrix
python evaluate.py

# Test inference on a single image
python predict.py --image test_photo.jpg

```

### Phase 3: Build the Pose Library

```bash
cd phase3_suggester
pip install -r requirements.txt

# Build the KNN index (outputs to pose_library/)
python build_pose_library.py

# Migrate models and library to the backend
cp -r pose_library/ ../phase4_backend/pose_library/
cp -r ../phase2_classifier/models/ ../phase4_backend/models/

```

### Phase 4: Application Deployment

**1. Start the Backend server:**

```bash
cd phase4_backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

```

**2. Start the Frontend client:**

```bash
cd phase4_frontend
npm install
npm run dev

```

The application will be available at `http://localhost:5173`.

## API Reference

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/analyse` | Accepts an image upload and returns scene classification, current keypoints, and top 3 pose suggestions. |
| `GET` | `/scenes` | Returns a list of all supported scene labels. |
| `GET` | `/health` | Backend liveness probe. |

## Results

*(Below are screenshots demonstrating the pipeline from initial upload to the final skeleton overlay.)*

### 1. Initial Image Upload


*User uploads an image via the React interface.*
<img width="1919" height="860" alt="Screenshot 2026-05-19 184544" src="https://github.com/user-attachments/assets/ee24195a-9090-4e3d-bb14-4426bce2354a" />


### 2. Scene Detection & Keypoint Extraction


*The backend identifies the scene (e.g., "Nature / Full Body") and extracts the user's current skeletal landmarks.*
<img width="1919" height="935" alt="Screenshot 2026-05-19 022704" src="https://github.com/user-attachments/assets/c1e9fd78-5abc-4ffa-9d67-09d6b5a45262" />


### 3. Pose Recommendations Applied


*The system overlays the top 3 KNN-recommended poses onto the original image for the user to match.*
<img width="1900" height="867" alt="Screenshot 2026-05-19 184653" src="https://github.com/user-attachments/assets/b527fc2c-b01f-4ba6-9243-d9beb677f7cb" />
<img width="541" height="854" alt="Screenshot 2026-05-19 184729" src="https://github.com/user-attachments/assets/343a2cd3-375a-42a8-99b4-16a5b33fb74c" />


### 4. Overall Architecture
<img width="1402" height="1122" alt="ChatGPT Image May 19, 2026, 10_47_29 AM" src="https://github.com/user-attachments/assets/12d2363e-b03e-41af-accf-62c44fa0f56c" />

---

```

```
