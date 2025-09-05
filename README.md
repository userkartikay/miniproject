# OptiCount - Smart Camera Inspection System

A real-time object detection and quality control system using computer vision for industrial inspection.

## Features

- **Real-time Detection**: Live camera feed with object detection and size measurement
- **Quality Control**: Automated defect detection based on dimensional tolerances
- **Statistics Dashboard**: Real-time tracking of total count, defects, quality rate, and efficiency
- **Industrial UI**: Professional dark theme interface designed for factory environments
- **Flexible Camera Support**: Works with USB cameras, IP cameras, and mobile camera apps

## Technology Stack

- **Backend**: Python, Flask, OpenCV
- **Frontend**: HTML5, CSS3, JavaScript
- **Computer Vision**: OpenCV with advanced image processing (CLAHE, bilateral filtering, perspective correction)

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the backend server:
   ```bash
   python backend.py
   ```

3. Open `frontend/index.html` in a web browser

4. Click "Start Inspection" to begin real-time detection

## Configuration

- Reference object dimensions: 5.0×3.0cm (configurable in `model.py`)
- Detection tolerance: ±0.3cm (configurable)
- Camera index: Auto-detect or configure in `backend.py`

## Architecture

- `model.py`: Computer vision processing and statistics engine
- `backend.py`: Flask API server and camera interface
- `frontend/`: Web-based user interface with real-time updates