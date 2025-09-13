import cv2
import numpy as np
import math
import time
from datetime import datetime

class OptiCountModel:
    def __init__(self):
        # Reference object dimensions (physical frame/container)
        self.reference_object_w = 22    # cm (frame width)
        self.reference_object_h = 28.23 # cm (frame height)
        
        # Target object dimensions (objects we want to measure inside the frame)
        self.target_object_w = 5.0      # cm (expected object width)
        self.target_object_h = 3.0      # cm (expected object height)
        self.tolerance = 0.3            # cm tolerance for defect detection
        self.scale = 10                 # Scale factor for perspective transformation
        
        # Statistics tracking
        self.total_count = 0
        self.defect_count = 0
        self.pass_count = 0
        self.detection_log = []
        self.processing_times = []
        
        # Image processing parameters
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.kernel = np.ones((3,3), np.uint8)
        
        # Frame storage for dual feeds
        self.contour_frame = None
        self.measurement_frame = None

    def biggest_contour(self, contours):
        """Find the biggest rectangular contour (reference object)"""
        biggest = np.array([])
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= 1000:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest

    def order_contour(self, pts):
        """Order contour points: top-left, top-right, bottom-right, bottom-left"""
        if pts.size != 0:
            pts = pts.reshape(4, 2)
            s = pts.sum(axis=1)
            d = np.diff(pts, axis=1)
            tl = pts[np.argmin(s)]
            br = pts[np.argmax(s)]
            tr = pts[np.argmin(d)]
            bl = pts[np.argmax(d)]
            return np.array([tl, tr, br, bl], dtype=np.int32)
        return pts

    def distance(self, p1, p2):
        """Calculate distance between two points"""
        return math.dist(p1, p2)

    def process_frame(self, frame):
        """Process frame and detect objects with measurements"""
        start_time = time.time()
        detected_objects = []
        
        if frame is None:
            return frame, detected_objects
            
        # Create copies for different visualizations
        contour_frame = frame.copy()  # For contour detection visualization
        measurement_frame = frame.copy()  # For measurement visualization
        
        # Image preprocessing (your original logic)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe_img = self.clahe.apply(gray)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        edge = cv2.Canny(blurred, 50, 100)
        
        # Morphological operations
        eroded = cv2.erode(edge, self.kernel, iterations=1)
        dilated = cv2.dilate(eroded, self.kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        # Find the biggest contour (reference object) - THIS IS THE KEY
        big_contour = self.biggest_contour(sorted_contours)
        
        # Only draw the reference contour on contour frame (if found)
        if big_contour.size != 0:
            cv2.drawContours(contour_frame, [big_contour], -1, (0, 255, 0), 3)
            
            # Add contour info
            cv2.putText(contour_frame, "Reference Contour Found", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # No reference contour found
            cv2.putText(contour_frame, "No Reference Contour Found", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        if big_contour.size != 0:
            # Order the contour points
            ordered = self.order_contour(big_contour)
            
            # Define destination points for perspective transformation
            dst_pts = np.array([
                [0, 0], 
                [self.reference_object_w * self.scale, 0], 
                [self.reference_object_w * self.scale, self.reference_object_h * self.scale], 
                [0, self.reference_object_h * self.scale]
            ], dtype=np.float32)
            
            # Perspective transformation
            M = cv2.getPerspectiveTransform(ordered.astype(np.float32), dst_pts)
            warped = cv2.warpPerspective(frame, M, 
                                       (int(self.reference_object_w * self.scale), 
                                        int(self.reference_object_h * self.scale)))
            
            # Calculate pixels per unit
            px_per_unit_w = warped.shape[1] / self.reference_object_w if self.reference_object_w != 0 else 0
            px_per_unit_h = warped.shape[0] / self.reference_object_h if self.reference_object_h != 0 else 0
            
            # Find objects in warped image (INSIDE the reference contour)
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            
            # Use different edge detection for objects inside the reference
            warped_blurred = cv2.GaussianBlur(warped_gray, (5, 5), 0)
            warped_edge = cv2.Canny(warped_blurred, 30, 80)  # Lower thresholds for inner objects
            
            # Morphological operations to clean up
            kernel_small = np.ones((2,2), np.uint8)
            warped_edge = cv2.morphologyEx(warped_edge, cv2.MORPH_CLOSE, kernel_small)
            
            contours_warped, _ = cv2.findContours(warped_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours - look for objects that are reasonable size but not the whole frame
            valid_contours = []
            min_area = 100  # Minimum area for a valid object
            max_area = (warped.shape[0] * warped.shape[1]) * 0.8  # Max 80% of frame
            
            for contour in contours_warped:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    # Additional filtering: check if it's roughly rectangular
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                    if len(approx) >= 4:  # At least 4 corners (rectangular-ish)
                        valid_contours.append(contour)
            
            # Process each detected object INSIDE the reference contour
            for i, contour in enumerate(valid_contours):
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate real-world dimensions
                w_obj = w / px_per_unit_w if px_per_unit_w != 0 else 0
                h_obj = h / px_per_unit_h if px_per_unit_h != 0 else 0
                
                # Calculate deviations from TARGET object (not reference frame)
                w_diff = abs(w_obj - self.target_object_w)
                h_diff = abs(h_obj - self.target_object_h)
                
                # Determine if object is defective
                is_defect = w_diff > self.tolerance or h_diff > self.tolerance
                status = "PASS" if not is_defect else "DEFECT"
                
                # Create object data
                obj_data = {
                    'width_cm': round(w_obj, 2),
                    'height_cm': round(h_obj, 2),
                    'target_width': self.target_object_w,
                    'target_height': self.target_object_h,
                    'reference_width': self.reference_object_w,
                    'reference_height': self.reference_object_h,
                    'w_diff': round(w_diff, 2),
                    'h_diff': round(h_diff, 2),
                    'is_defect': is_defect,
                    'area': cv2.contourArea(contour),
                    'aspect_ratio': w_obj / h_obj if h_obj != 0 else 0
                }
                
                detected_objects.append(obj_data)
                
                # Transform contour back to original image coordinates for accurate drawing
                contour_orig = cv2.perspectiveTransform(contour.astype(np.float32).reshape(-1, 1, 2), 
                                                      cv2.invert(M)[1])
                
                # Add simple counter on contour frame
                cv2.putText(contour_frame, f"Objects Found: {i+1}", 
                           (10, 150 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           (255, 0, 255) if not is_defect else (0, 0, 255), 2)
                
                # Draw on measurement frame with detailed information
                x_orig, y_orig, w_orig, h_orig = cv2.boundingRect(contour_orig.astype(np.int32))
                
                # Color based on defect status
                color = (0, 255, 0) if not is_defect else (0, 0, 255)
                status_color = (0, 255, 0) if not is_defect else (0, 0, 255)
                
                # Draw rectangle around detected object
                cv2.rectangle(measurement_frame, (x_orig, y_orig), (x_orig+w_orig, y_orig+h_orig), color, 3)
                
                # Draw detailed measurement popup-style overlay
                popup_height = 130
                popup_width = 280
                popup_x = min(x_orig, measurement_frame.shape[1] - popup_width - 10)
                popup_y = max(y_orig - popup_height - 10, 10)
                
                # Create semi-transparent background for popup
                overlay = measurement_frame.copy()
                cv2.rectangle(overlay, (popup_x, popup_y), 
                            (popup_x + popup_width, popup_y + popup_height), 
                            (50, 50, 50), -1)
                cv2.addWeighted(overlay, 0.85, measurement_frame, 0.15, 0, measurement_frame)
                
                # Add border to popup
                cv2.rectangle(measurement_frame, (popup_x, popup_y), 
                            (popup_x + popup_width, popup_y + popup_height), 
                            status_color, 3)
                
                # Add measurement text lines with better formatting
                text_x = popup_x + 15
                text_y = popup_y + 25
                line_height = 18
                
                # Status (PASS/DEFECT) - Large and prominent
                cv2.putText(measurement_frame, f"STATUS: {status}", 
                           (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                text_y += line_height + 2
                
                # Measured dimensions
                cv2.putText(measurement_frame, f"Measured: {w_obj:.2f} x {h_obj:.2f} cm", 
                           (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                text_y += line_height
                
                # Target dimensions (what we expect)
                cv2.putText(measurement_frame, f"Target: {self.target_object_w:.2f} x {self.target_object_h:.2f} cm", 
                           (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                text_y += line_height
                
                # Deviation from target
                cv2.putText(measurement_frame, f"Deviation: W±{w_diff:.2f}, H±{h_diff:.2f} cm", 
                           (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                text_y += line_height
                
                # Tolerance
                cv2.putText(measurement_frame, f"Tolerance: ±{self.tolerance:.2f} cm", 
                           (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                text_y += line_height
                
                # Aspect ratio
                cv2.putText(measurement_frame, f"Aspect: {obj_data['aspect_ratio']:.2f}", 
                           (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Draw reference contour prominently on contour frame
            cv2.drawContours(contour_frame, [big_contour], -1, (0, 255, 0), 4)
            # Draw reference contour subtly on measurement frame  
            cv2.drawContours(measurement_frame, [big_contour], -1, (100, 100, 100), 2)
            
            # Add reference object info to contour frame
            cv2.putText(contour_frame, f"Frame: {self.reference_object_w}x{self.reference_object_h}cm", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(contour_frame, f"Target: {self.target_object_w}x{self.target_object_h}cm", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(contour_frame, f"Objects Inside: {len(detected_objects)}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add labels to identify the frames
            cv2.putText(contour_frame, "REFERENCE CONTOUR DETECTION", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(measurement_frame, "OBJECTS INSIDE CONTOUR", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show detection count on measurement frame
            cv2.putText(measurement_frame, f"Objects Detected: {len(detected_objects)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # No reference contour found, add status message
            cv2.putText(contour_frame, "NO REFERENCE CONTOUR FOUND", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(measurement_frame, "NO REFERENCE CONTOUR FOUND", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Record processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:  # Keep last 100 measurements
            self.processing_times.pop(0)
        
        # Store both frames for dual feed access
        self.contour_frame = contour_frame
        self.measurement_frame = measurement_frame
        
        return measurement_frame, detected_objects

    def add_detection(self, detected_objects):
        """Add detected objects to statistics and log"""
        for obj in detected_objects:
            self.total_count += 1
            
            if obj['is_defect']:
                self.defect_count += 1
            else:
                self.pass_count += 1
            
            # Add to detection log with timestamp
            log_entry = obj.copy()
            log_entry['timestamp'] = datetime.now().strftime("%H:%M:%S")
            self.detection_log.append(log_entry)
            
            # Keep only last 50 detections in log
            if len(self.detection_log) > 50:
                self.detection_log.pop(0)

    def get_statistics(self):
        """Get current statistics"""
        quality_rate = self.calculate_quality_rate()
        efficiency = self.calculate_efficiency()
        
        return {
            'total_count': self.total_count,
            'defect_count': self.defect_count,
            'pass_count': self.pass_count,
            'quality_rate': quality_rate,
            'efficiency': efficiency
        }

    def calculate_quality_rate(self):
        """Calculate quality rate percentage"""
        if self.total_count == 0:
            return 100.0
        return round((self.pass_count / self.total_count) * 100, 1)

    def calculate_efficiency(self):
        """Calculate processing efficiency based on frame processing time"""
        if not self.processing_times:
            return 95.0
        
        avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        target_time = 1.0 / 30.0  # Target 30 FPS
        
        if avg_processing_time <= target_time:
            return 100.0
        else:
            efficiency = (target_time / avg_processing_time) * 100
            return max(10.0, min(100.0, round(efficiency, 1)))

    def reset_statistics(self):
        """Reset all statistics"""
        self.total_count = 0
        self.defect_count = 0
        self.pass_count = 0
        self.detection_log.clear()
        self.processing_times.clear()

    def get_detection_log(self):
        """Get recent detection log"""
        return self.detection_log.copy()

    def calibrate(self, width, height, tolerance):
        """Calibrate reference frame dimensions"""
        self.reference_object_w = width
        self.reference_object_h = height
        self.tolerance = tolerance
        print(f"Calibrated reference frame: {width}x{height}cm ±{tolerance}cm")

    def calibrate_target(self, width, height, tolerance=None):
        """Calibrate target object dimensions (objects inside the frame)"""
        self.target_object_w = width
        self.target_object_h = height
        if tolerance is not None:
            self.tolerance = tolerance
        print(f"Calibrated target object: {width}x{height}cm ±{self.tolerance}cm")

    def update_settings(self, **kwargs):
        """Update processing settings"""
        if 'reference_object_w' in kwargs:
            self.reference_object_w = kwargs['reference_object_w']
        if 'reference_object_h' in kwargs:
            self.reference_object_h = kwargs['reference_object_h']
        if 'target_object_w' in kwargs:
            self.target_object_w = kwargs['target_object_w']
        if 'target_object_h' in kwargs:
            self.target_object_h = kwargs['target_object_h']
        if 'tolerance' in kwargs:
            self.tolerance = kwargs['tolerance']
        print(f"Settings updated: {kwargs}")
