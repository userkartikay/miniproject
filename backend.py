from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
import json
import threading
import time
from datetime import datetime
from model import OptiCountModel

app = Flask(__name__)
CORS(app)

class OptiCountBridge:
    def __init__(self):
        self.is_running = False
        self.cap = None
        self.detection_log = []
        self.total_count = 0
        self.defect_count = 0
        self.last_detection_time = 0
        
        # Initialize the model
        self.model = OptiCountModel()
        
    def start_camera(self):
        try:
            self.cap = cv2.VideoCapture(1)  # Try camera 1 first
            if not self.cap.isOpened():
                print("Camera 1 not available, trying camera 0")
                self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                print("No camera available")
                return False
                
            # Set camera properties for better quality
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            print("Camera started with improved settings")
            return True
        except Exception as e:
            print(f"Camera error: {e}")
            return False
    
    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def get_frame(self):
        if not self.is_running or not self.cap:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # Add FPS display
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame, f'FPS: {int(fps)}', (frame.shape[1]-120, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 156), 2)
        
        # Process frame using the model
        processed_frame, detected_objects = self.model.process_frame(frame)
        
        # Handle detection logging and statistics update (FIXED LOGIC)
        current_time = time.time()
        
        # If objects are detected, always update statistics but rate-limit logging
        if detected_objects:
            # Always update model statistics for each detection
            self.model.add_detection(detected_objects)
            
            # Rate-limit detailed logging to avoid spam
            if current_time - self.last_detection_time > 1.5:
                # Keep backend detection log for compatibility
                for obj in detected_objects:
                    detection = {
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'width_cm': obj['width_cm'],
                        'height_cm': obj['height_cm'],
                        'reference_width': self.model.reference_object_w,
                        'reference_height': self.model.reference_object_h,
                        'w_diff': obj['w_diff'],
                        'h_diff': obj['h_diff'],
                        'is_defect': obj['is_defect'],
                        'area': obj['area'],
                        'aspect_ratio': obj['aspect_ratio']
                    }
                    
                    self.detection_log.append(detection)
                    if len(self.detection_log) > 50:
                        self.detection_log = self.detection_log[-50:]
                    
                    # Legacy counters for compatibility
                    self.total_count += 1
                    if obj['is_defect']:
                        self.defect_count += 1
                
                self.last_detection_time = current_time
        
        return processed_frame
    
    def get_detection_data(self):
        # Get statistics from model and combine with detection log
        model_stats = self.model.get_statistics()
        
        result = {
            'total_count': model_stats['total_count'],
            'defect_count': model_stats['defect_count'],
            'quality_rate': model_stats['quality_rate'],
            'efficiency': model_stats['efficiency'],
            'detections': self.detection_log.copy()
        }
        
        return result

# Global system instance
opti_system = OptiCountBridge()

def generate_frames():
    while opti_system.is_running:
        frame = opti_system.get_frame()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)  # ~30 FPS

@app.route('/api/start', methods=['POST'])
def start_inspection():
    try:
        if opti_system.start_camera():
            return jsonify({'status': 'success', 'message': 'Inspection started'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to start camera'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/stop', methods=['POST'])
def stop_inspection():
    try:
        opti_system.stop_camera()
        return jsonify({'status': 'success', 'message': 'Inspection stopped'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/data', methods=['GET'])
def get_detection_data():
    return jsonify(opti_system.get_detection_data())

@app.route('/api/calibrate', methods=['POST'])
def calibrate_reference():
    """Set reference object dimensions with high precision"""
    try:
        data = request.json
        width = float(data.get('width', 5.0))
        height = float(data.get('height', 3.0))
        tolerance = float(data.get('tolerance', 0.3))
        
        opti_system.model.calibrate(width, height, tolerance)
        
        return jsonify({
            'status': 'success', 
            'message': f'Reference object: {width}x{height}cm ±{tolerance}cm'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/settings', methods=['POST'])
def update_settings():
    try:
        data = request.json
        opti_system.model.update_settings(**data)
        
        return jsonify({'status': 'success', 'message': 'Settings updated'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/reset_stats', methods=['POST'])
def reset_statistics():
    try:
        opti_system.model.reset_statistics()
        opti_system.detection_log = []
        opti_system.total_count = 0
        opti_system.defect_count = 0
        return jsonify({'status': 'success', 'message': 'Statistics reset successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/video_feed')
def video_feed():
    if not opti_system.is_running:
        return jsonify({'error': 'Camera not started'}), 400
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <h1>OptiCount Backend Bridge</h1>
    <p>Flask backend bridging frontend and OpenCV model</p>
    <p>Open frontend/index.html to use the system</p>
    '''

if __name__ == '__main__':
    print("Starting OptiCount Backend Bridge...")
    print("Model-Bridge Architecture Active")
    print("Server running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
