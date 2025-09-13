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
        self.last_detection_time = 0
        
        # Frame storage for dual feeds
        self.raw_frame = None
        self.processed_frame = None
        
        # Initialize the model
        self.model = OptiCountModel()
        
    def start_camera(self):
        """Start camera with robust fallback logic"""
        camera_indices = [1, 0]  # Try camera 1 first, then camera 0
        
        for camera_index in camera_indices:
            try:
                print(f"Attempting to start camera {camera_index}...")
                self.cap = cv2.VideoCapture(camera_index)
                
                # Give camera time to initialize
                time.sleep(0.5)
                
                if self.cap.isOpened():
                    # Test if camera can actually read frames
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None and test_frame.shape[0] > 0 and test_frame.shape[1] > 0:
                        print(f"Camera {camera_index} successfully initialized")
                        
                        # Set camera properties for better quality
                        try:
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower resolution for stability
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            self.cap.set(cv2.CAP_PROP_FPS, 30)
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Reduce buffer to prevent lag
                        except Exception as prop_e:
                            print(f"Warning: Could not set camera properties: {prop_e}")
                        
                        # Verify settings
                        try:
                            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                            print(f"Camera resolution: {int(actual_width)}x{int(actual_height)}")
                        except:
                            print("Could not read camera properties")
                        
                        self.is_running = True
                        return True
                    else:
                        print(f"Camera {camera_index} opened but cannot read valid frames")
                        self.cap.release()
                        self.cap = None
                else:
                    print(f"Camera {camera_index} failed to open")
                    if self.cap:
                        self.cap.release()
                        self.cap = None
                        
            except Exception as e:
                print(f"Error with camera {camera_index}: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
        
        print("No cameras available or all cameras failed")
        return False
    
    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def get_frame(self):
        """Get frame with robust error handling"""
        if not self.is_running or not self.cap:
            print("DEBUG: Camera not running or not available")
            return None
            
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("DEBUG: Failed to read frame from camera")
                return None
            
            # Validate frame dimensions to prevent OpenCV matrix errors
            if frame.shape[0] <= 0 or frame.shape[1] <= 0 or len(frame.shape) != 3:
                print("DEBUG: Invalid frame dimensions")
                return None
            
            # Add FPS display
            try:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                cv2.putText(frame, f'FPS: {int(fps)}', (frame.shape[1]-120, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 156), 2)
            except:
                pass  # Continue without FPS display if it fails
            
            # Process frame using the model
            try:
                processed_frame, detected_objects = self.model.process_frame(frame)
                
                # Store frames for dual feed access
                self.raw_frame = self.model.contour_frame if self.model.contour_frame is not None else frame.copy()
                self.processed_frame = processed_frame.copy() if processed_frame is not None else frame.copy()
                
            except Exception as e:
                print(f"DEBUG: Model processing error: {e}")
                # Use original frame if processing fails
                self.raw_frame = frame.copy()
                self.processed_frame = frame.copy()
                detected_objects = []
                
        except Exception as e:
            print(f"DEBUG: Camera read error: {e}")
            # Stop camera to prevent infinite error loop
            self.stop_camera()
            return None
        
        # Handle detection logging and statistics update (FIXED LOGIC)
        current_time = time.time()
        
        # If objects are detected, always update statistics but rate-limit logging
        if detected_objects:
            print(f"DEBUG: Detected {len(detected_objects)} objects")  # Debug log
            
            # Always update model statistics for each detection
            self.model.add_detection(detected_objects)
            
            # Rate-limit detailed logging to avoid spam
            if current_time - self.last_detection_time > 1.5:
                print(f"DEBUG: Adding to detection log")  # Debug log
                # Keep backend detection log for compatibility
                for i, obj in enumerate(detected_objects):
                    print(f"DEBUG: Object {i+1}: {obj['width_cm']:.2f}x{obj['height_cm']:.2f}cm, Defect: {obj['is_defect']}")
                    
                    detection = {
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'width_cm': obj['width_cm'],
                        'height_cm': obj['height_cm'],
                        'reference_width': self.model.reference_object_w,
                        'reference_height': self.model.reference_object_h,
                        'target_width': self.model.target_object_w,
                        'target_height': self.model.target_object_h,
                        'tolerance': self.model.tolerance,
                        'w_diff': obj['w_diff'],
                        'h_diff': obj['h_diff'],
                        'is_defect': obj['is_defect'],
                        'area': obj['area'],
                        'aspect_ratio': obj['aspect_ratio']
                    }
                    
                    self.detection_log.append(detection)
                    if len(self.detection_log) > 50:
                        self.detection_log = self.detection_log[-50:]
                
                self.last_detection_time = current_time
        
        return processed_frame
    
    def get_raw_frame(self):
        """Get raw camera frame with contours"""
        return self.raw_frame if self.raw_frame is not None else None
        
    def get_processed_frame(self):
        """Get processed frame with measurements"""
        return self.processed_frame if self.processed_frame is not None else None
    
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

def generate_raw_frames():
    """Generate raw camera frames with contour detection"""
    while True:  # Continue even if camera is stopped
        try:
            if opti_system.is_running:
                opti_system.get_frame()  # Update frames
                frame = opti_system.get_raw_frame()
                if frame is not None:
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    else:
                        # Send error frame if encoding fails
                        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(placeholder, "Frame Encoding Error", (180, 240), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        ret, buffer = cv2.imencode('.jpg', placeholder)
                        if ret:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # Send a placeholder frame if no camera frame available
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "No Camera Feed", (200, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(placeholder, "Press START to retry", (180, 280), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', placeholder)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # Camera not running - show stopped message
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Camera Stopped", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
                cv2.putText(placeholder, "Press START to begin", (180, 280), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', placeholder)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Raw frame generation error: {e}")
            # Send error frame
            try:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Stream Error", (220, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', placeholder)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except:
                pass
        time.sleep(0.1)  # Slower rate when error/stopped

def generate_processed_frames():
    """Generate processed frames with dimension measurements"""
    while True:  # Continue even if camera is stopped
        try:
            if opti_system.is_running:
                opti_system.get_frame()  # Update frames
                frame = opti_system.get_processed_frame()
                if frame is not None:
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    else:
                        # Send error frame if encoding fails
                        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(placeholder, "Frame Encoding Error", (180, 240), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        ret, buffer = cv2.imencode('.jpg', placeholder)
                        if ret:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # Send a placeholder frame if no camera frame available
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "No Camera Feed", (200, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(placeholder, "Press START to retry", (180, 280), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', placeholder)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # Camera not running - show stopped message
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Camera Stopped", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
                cv2.putText(placeholder, "Press START to begin", (180, 280), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', placeholder)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Processed frame generation error: {e}")
            # Send error frame
            try:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Stream Error", (220, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', placeholder)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except:
                pass
        time.sleep(0.1)  # Slower rate when error/stopped

@app.route('/api/start', methods=['POST'])
def start_inspection():
    try:
        if opti_system.start_camera():
            return jsonify({'status': 'success', 'message': 'Inspection started'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to start camera. No cameras available.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/camera_status', methods=['GET'])
def camera_status():
    """Get camera status information"""
    try:
        status = {
            'is_running': opti_system.is_running,
            'camera_available': opti_system.cap is not None and opti_system.cap.isOpened(),
            'camera_index': None,
            'resolution': None,
            'fps': None
        }
        
        if opti_system.cap and opti_system.cap.isOpened():
            # Try to get camera properties
            try:
                status['resolution'] = {
                    'width': int(opti_system.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(opti_system.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                }
                status['fps'] = int(opti_system.cap.get(cv2.CAP_PROP_FPS))
            except:
                pass
        
        return jsonify(status)
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
    result = opti_system.get_detection_data()
    print(f"DEBUG: Sending detection data - Total: {result.get('total_count', 0)}, Detections: {len(result.get('detections', []))}")
    return jsonify(result)

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
        return jsonify({'status': 'success', 'message': 'Statistics reset successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/video_feed')
def video_feed():
    if not opti_system.is_running:
        return jsonify({'error': 'Camera not started'}), 400
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_raw')
def video_feed_raw():
    """Raw camera feed with contour detection"""
    if not opti_system.is_running:
        return jsonify({'error': 'Camera not started'}), 400
    
    return Response(generate_raw_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_processed')
def video_feed_processed():
    """Processed feed with dimension measurements"""
    if not opti_system.is_running:
        return jsonify({'error': 'Camera not started'}), 400
    
    return Response(generate_processed_frames(),
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
