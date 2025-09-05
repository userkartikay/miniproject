import cv2
import numpy as np
import math
from datetime import datetime
import time

class OptiCountModel:
    def __init__(self):
        # Calibrated settings for accuracy
        self.w_real = 22.0  # cm - reference surface width
        self.h_real = 28.23  # cm - reference surface height
        self.scale = 10
        
        # Reference object dimensions
        self.reference_object_w = 5.0  # cm - width of reference object
        self.reference_object_h = 3.0  # cm - height of reference object
        self.tolerance = 0.3  # cm tolerance for defects
        
        # Detection parameters for accuracy
        self.min_area_pixels = 400  # Minimum object area in pixels
        self.max_area_pixels = 15000  # Maximum object area in pixels
        self.min_size_cm = 0.8  # Minimum object size in cm
        self.max_size_cm = 12.0  # Maximum object size in cm
        
        # Statistics tracking (NEW)
        self.total_count = 0
        self.defect_count = 0
        self.pass_count = 0
        self.session_start_time = time.time()
        self.processing_times = []

    def biggest_contour(self, contor):
        biggest = np.array([])
        max_area = 0
        for i in contor:
            area = cv2.contourArea(i)
            if area >= 1500:  # Higher threshold for reference surface
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.015*peri, True)  # More precise approximation
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest

    def order_contour(self, pts):
        if pts.size != 0:
            pts = pts.reshape(4, 2)
            s = pts.sum(axis=1)
            d = np.diff(pts, axis=1)
            tl = pts[np.argmin(s)]
            br = pts[np.argmax(s)]
            tr = pts[np.argmin(d)]
            bl = pts[np.argmax(d)]
            return np.array([tl, tr, br, bl], dtype=np.int32)

    def distance(self, p1, p2):
        return math.dist(p1, p2)

    def preprocess_frame(self, frame):
        """Advanced preprocessing for accurate detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Bilateral filter to preserve edges while smoothing
        filtered = cv2.bilateralFilter(blurred, 9, 75, 75)
        
        return filtered

    def find_reference_surface(self, processed_frame):
        """Find the reference surface with high accuracy"""
        # Edge detection with optimized parameters
        edges = cv2.Canny(processed_frame, 30, 90, apertureSize=3, L2gradient=True)
        
        # Morphological operations to clean up edges
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        return self.biggest_contour(contours)

    def detect_objects_in_warped(self, warped_frame):
        """Detect objects in the warped perspective with high accuracy"""
        # Convert to grayscale
        warped_gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive preprocessing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
        warped_enhanced = clahe.apply(warped_gray)
        
        # Gaussian blur
        warped_blurred = cv2.GaussianBlur(warped_enhanced, (3, 3), 0)
        
        # Adaptive threshold for better object separation
        adaptive_thresh = cv2.adaptiveThreshold(warped_blurred, 255, 
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY_INV, 11, 2)
        
        # Edge detection with fine-tuned parameters
        edges = cv2.Canny(warped_blurred, 20, 60, apertureSize=3)
        
        # Combine adaptive threshold and edges
        combined = cv2.bitwise_or(adaptive_thresh, edges)
        
        # Clean up with morphological operations
        kernel_small = np.ones((2,2), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_small)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours, warped_enhanced

    def process_frame(self, frame):
        """Main detection processing function"""
        start_time = time.time()
        detected_objects = []
        
        # Add system status (keep only basic info on camera)
        cv2.putText(frame, "OptiCount Detection", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Preprocess the frame
        processed = self.preprocess_frame(frame)
        
        # Find reference surface
        big_contour = self.find_reference_surface(processed)
        
        if big_contour.size != 0:
            # Found reference surface
            cv2.drawContours(frame, [big_contour], -1, (0, 255, 0), 3)
            cv2.putText(frame, "Reference Surface Found", (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            ordered = self.order_contour(big_contour)
            
            # Perspective transformation
            dst_pts = np.array([[0, 0], [self.w_real*self.scale, 0], 
                              [self.w_real*self.scale, self.h_real*self.scale], 
                              [0, self.h_real*self.scale]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(ordered.astype(np.float32), dst_pts)
            warped = cv2.warpPerspective(frame, M, (int(self.w_real*self.scale), int(self.h_real*self.scale)))
            
            # Calculate pixel-to-cm conversion factors
            px_per_unit_w = warped.shape[1] / self.w_real
            px_per_unit_h = warped.shape[0] / self.h_real
            
            # Detect objects in warped area
            contours_warped, warped_processed = self.detect_objects_in_warped(warped)
            
            objects_found = 0
            
            for contour in contours_warped:
                area = cv2.contourArea(contour)
                
                # Filter by area
                if area < self.min_area_pixels or area > self.max_area_pixels:
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate dimensions in cm
                w_obj = w / px_per_unit_w
                h_obj = h / px_per_unit_h
                
                # Filter by size in cm
                if (w_obj < self.min_size_cm or h_obj < self.min_size_cm or 
                    w_obj > self.max_size_cm or h_obj > self.max_size_cm):
                    continue
                
                # Filter out objects that are too similar to the reference surface
                if abs(w_obj - self.w_real) < 3.0 and abs(h_obj - self.h_real) < 3.0:
                    continue
                
                # Check aspect ratio to filter noise
                aspect_ratio = w_obj / h_obj if h_obj > 0 else 0
                if aspect_ratio < 0.2 or aspect_ratio > 5.0:  # Reasonable aspect ratios
                    continue
                
                objects_found += 1
                
                # Calculate deviations from reference object
                w_diff = abs(w_obj - self.reference_object_w)
                h_diff = abs(h_obj - self.reference_object_h)
                is_defect = w_diff > self.tolerance or h_diff > self.tolerance
                
                # Visualization
                color = (0, 0, 255) if is_defect else (0, 255, 0)
                status = "DEFECT" if is_defect else "PASS"
                
                # Draw on warped view
                cv2.rectangle(warped, (x, y), (x+w, y+h), color, 2)
                cv2.putText(warped, f"{status}", (x, y-25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(warped, f"{w_obj:.2f}x{h_obj:.2f}cm", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Draw center point
                center = (x + w//2, y + h//2)
                cv2.circle(warped, center, 3, color, -1)
                
                # Store object info
                obj_info = {
                    'width_cm': w_obj,
                    'height_cm': h_obj,
                    'w_diff': w_diff,
                    'h_diff': h_diff,
                    'is_defect': is_defect,
                    'area': area,
                    'aspect_ratio': aspect_ratio
                }
                detected_objects.append(obj_info)
                
                # Display on main frame
                info_y = 85 + objects_found * 20
                cv2.putText(frame, f"Obj{objects_found}: {w_obj:.2f}x{h_obj:.2f}cm ({status})", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.putText(frame, f"Objects: {objects_found}", (10, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        else:
            cv2.putText(frame, "Place reference surface in view", (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show current settings
        cv2.putText(frame, f"Ref: {self.reference_object_w}x{self.reference_object_h}cm ±{self.tolerance}cm", 
                   (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Track processing time for efficiency calculation
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]
        
        return frame, detected_objects

    def calibrate(self, width, height, tolerance):
        """Update reference object dimensions"""
        self.reference_object_w = width
        self.reference_object_h = height
        self.tolerance = tolerance

    def update_settings(self, **kwargs):
        """Update detection settings"""
        if 'reference_object_w' in kwargs:
            self.reference_object_w = kwargs['reference_object_w']
        if 'reference_object_h' in kwargs:
            self.reference_object_h = kwargs['reference_object_h']
        if 'tolerance' in kwargs:
            self.tolerance = kwargs['tolerance']
        if 'min_area_pixels' in kwargs:
            self.min_area_pixels = kwargs['min_area_pixels']
        if 'max_area_pixels' in kwargs:
            self.max_area_pixels = kwargs['max_area_pixels']

    def add_detection(self, detected_objects):
        """Add detection results to statistics"""
        for obj in detected_objects:
            self.total_count += 1
            if obj['is_defect']:
                self.defect_count += 1
            else:
                self.pass_count += 1

    def calculate_quality_rate(self):
        """Calculate quality rate as percentage of passed items"""
        if self.total_count == 0:
            return 100.0
        return (self.pass_count / self.total_count) * 100.0

    def calculate_efficiency(self):
        """Calculate system efficiency based on processing performance"""
        if not self.processing_times:
            return 95.0  # Default efficiency
        
        # Calculate based on processing speed
        avg_processing_time = sum(self.processing_times[-50:]) / min(len(self.processing_times), 50)
        
        # Target processing time is 0.05 seconds (20 FPS)
        target_time = 0.05
        efficiency = min(100, (target_time / avg_processing_time) * 100) if avg_processing_time > 0 else 100
        
        return min(100.0, max(75.0, efficiency))  # Keep between 75-100%

    def get_statistics(self):
        """Get comprehensive statistics for the frontend"""
        return {
            'total_count': self.total_count,
            'defect_count': self.defect_count,
            'pass_count': self.pass_count,
            'quality_rate': round(self.calculate_quality_rate(), 1),
            'efficiency': round(self.calculate_efficiency(), 1)
        }

    def reset_statistics(self):
        """Reset all statistics"""
        self.total_count = 0
        self.defect_count = 0
        self.pass_count = 0
        self.session_start_time = time.time()
        self.processing_times = []


# For standalone testing
if __name__ == "__main__":
    model = OptiCountModel()
    webcam = cv2.VideoCapture(1)
    
    while True:
        is_true, frames = webcam.read()
      
        if not is_true:
            break
        
        fps = webcam.get(cv2.CAP_PROP_FPS)
        cv2.putText(frames, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 156))
        
        processed_frame, detected_objects = model.process_frame(frames)
        cv2.imshow("OptiCount Detection", processed_frame)

        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
            
    webcam.release()
    cv2.destroyAllWindows()