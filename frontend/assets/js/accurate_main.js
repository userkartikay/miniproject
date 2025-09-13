// Accurate OptiCount Smart Inspection System - Enhanced JavaScript
class AccurateOptiCount {
    constructor() {
        this.baseURL = 'http://localhost:5000/api';
        this.videoURLRaw = 'http://localhost:5000/video_feed_raw';
        this.videoURLProcessed = 'http://localhost:5000/video_feed_processed';
        this.isRunning = false;
        this.dataUpdateInterval = null;
        
        // Reference object calibration
        this.referenceWidth = 5.0;  // cm
        this.referenceHeight = 3.0; // cm
        this.tolerance = 0.3;       // cm
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.updateDisplay();
        this.showSystemStatus('Ready for accurate inspection');
    }

    bindEvents() {
        // Main controls
        document.getElementById('startBtn')?.addEventListener('click', () => this.startInspection());
        document.getElementById('stopBtn')?.addEventListener('click', () => this.stopInspection());
        document.getElementById('calibrateBtn')?.addEventListener('click', () => this.showCalibrationModal());
        
        // Settings controls
        document.getElementById('saveSettings')?.addEventListener('click', () => this.saveSettings());
        document.getElementById('resetStats')?.addEventListener('click', () => this.resetStatistics());
        
        // Calibration controls
        document.getElementById('saveCalibration')?.addEventListener('click', () => this.saveCalibration());
        document.getElementById('closeModal')?.addEventListener('click', () => this.hideCalibrationModal());
    }

    async startInspection() {
        try {
            this.showSystemStatus('Starting accurate inspection...', 'warning');
            
            const response = await fetch(`${this.baseURL}/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.isRunning = true;
                this.showSystemStatus('Accurate inspection active', 'success');
                this.startVideoFeed();
                this.startDataUpdates();
                this.updateControlButtons();
                this.playNotificationSound('start');
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            console.error('Start error:', error);
            this.showSystemStatus('Failed to start inspection: ' + error.message, 'error');
        }
    }

    async stopInspection() {
        try {
            const response = await fetch(`${this.baseURL}/stop`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.isRunning = false;
                this.showSystemStatus('Inspection stopped', 'warning');
                this.stopVideoFeed();
                this.stopDataUpdates();
                this.updateControlButtons();
                this.playNotificationSound('stop');
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            console.error('Stop error:', error);
            this.showSystemStatus('Failed to stop inspection: ' + error.message, 'error');
        }
    }

    startVideoFeed() {
        // Start raw camera feed (contour detection)
        const videoElementRaw = document.getElementById('videoFeedRaw');
        if (videoElementRaw) {
            videoElementRaw.src = this.videoURLRaw + '?t=' + Date.now();
            videoElementRaw.style.display = 'block';
        }
        
        // Start processed camera feed (dimension measurement)
        const videoElementProcessed = document.getElementById('videoFeedProcessed');
        if (videoElementProcessed) {
            videoElementProcessed.src = this.videoURLProcessed + '?t=' + Date.now();
            videoElementProcessed.style.display = 'block';
        }
    }

    stopVideoFeed() {
        // Stop raw camera feed
        const videoElementRaw = document.getElementById('videoFeedRaw');
        if (videoElementRaw) {
            videoElementRaw.src = '';
            videoElementRaw.style.display = 'none';
        }
        
        // Stop processed camera feed
        const videoElementProcessed = document.getElementById('videoFeedProcessed');
        if (videoElementProcessed) {
            videoElementProcessed.src = '';
            videoElementProcessed.style.display = 'none';
        }
    }

    startDataUpdates() {
        this.stopDataUpdates(); // Clear any existing interval
        
        this.dataUpdateInterval = setInterval(async () => {
            await this.updateDetectionData();
        }, 800); // Faster updates for real-time feel
    }

    stopDataUpdates() {
        if (this.dataUpdateInterval) {
            clearInterval(this.dataUpdateInterval);
            this.dataUpdateInterval = null;
        }
    }

    async updateDetectionData() {
        if (!this.isRunning) return;
        
        try {
            const response = await fetch(`${this.baseURL}/data`);
            const data = await response.json();
            
            // Debug logging
            console.log('Detection data received:', data);
            
            // Update stats
            this.updateStatistics(data);
            
            // Update detection log
            this.updateDetectionLog(data.detections || []);
            
        } catch (error) {
            console.error('Data update error:', error);
        }
    }

    updateStatistics(data) {
        console.log('Updating statistics with:', data); // Debug log
        
        // Only update if we have valid data from the API
        if (data && typeof data === 'object') {
            if (data.total_count !== undefined) {
                console.log('Updating total count from', this.getCurrentDisplayValue('totalCount'), 'to', data.total_count);
                this.animateNumber('totalCount', data.total_count);
            }
            
            if (data.defect_count !== undefined) {
                console.log('Updating defect count from', this.getCurrentDisplayValue('defectCount'), 'to', data.defect_count);
                this.animateNumber('defectCount', data.defect_count);
            }
            
            if (data.quality_rate !== undefined) {
                console.log('Updating quality rate from', this.getCurrentDisplayValue('qualityRate', '%'), 'to', data.quality_rate + '%');
                this.animateNumber('qualityRate', data.quality_rate, '%');
                this.updateQualityIndicator(data.quality_rate);
            }
            
            if (data.efficiency !== undefined) {
                console.log('Updating efficiency from', this.getCurrentDisplayValue('efficiency', '%'), 'to', data.efficiency + '%');
                this.animateNumber('efficiency', data.efficiency, '%');
            }
        } else {
            console.warn('Invalid or empty statistics data received:', data);
        }
    }

    getCurrentDisplayValue(elementId, suffix = '') {
        const element = document.getElementById(elementId);
        if (!element || !element.textContent) return 0;
        
        let value = element.textContent.replace(suffix, '');
        return parseFloat(value) || 0;
    }

    updateDetectionLog(detections) {
        const logContainer = document.getElementById('detectionLog');
        if (!logContainer) return;
        
        // Clear existing log
        logContainer.innerHTML = '';
        
        if (detections.length === 0) {
            logContainer.innerHTML = '<div class="log-entry no-data">No detections yet...</div>';
            return;
        }
        
        // Show recent detections (last 10)
        const recentDetections = detections.slice(-10).reverse();
        
        recentDetections.forEach(detection => {
            const logEntry = document.createElement('div');
            logEntry.className = `log-entry ${detection.is_defect ? 'defect' : 'pass'}`;
            
            const statusIcon = detection.is_defect ? '❌' : '✅';
            const status = detection.is_defect ? 'DEFECT' : 'PASS';
            
            // Use tolerance from detection data if available, otherwise use current tolerance
            const currentTolerance = detection.tolerance !== undefined ? detection.tolerance : this.tolerance;
            
            // Determine pass/fail status based on tolerance
            const wExceedsTolerance = detection.w_diff > currentTolerance;
            const hExceedsTolerance = detection.h_diff > currentTolerance;
            
            let toleranceStatus = '';
            if (wExceedsTolerance && hExceedsTolerance) {
                toleranceStatus = ' (Both dimensions exceed tolerance)';
            } else if (wExceedsTolerance) {
                toleranceStatus = ' (Width exceeds tolerance)';
            } else if (hExceedsTolerance) {
                toleranceStatus = ' (Height exceeds tolerance)';
            }
            
            logEntry.innerHTML = `
                <div class="log-time">${detection.timestamp}</div>
                <div class="log-status">${statusIcon} ${status}${toleranceStatus}</div>
                <div class="log-details">
                    <div class="size-info">
                        Measured: ${detection.width_cm?.toFixed(2)} × ${detection.height_cm?.toFixed(2)} cm
                    </div>
                    <div class="target-info">
                        Target: ${(detection.target_width || this.referenceWidth)?.toFixed(2)} × ${(detection.target_height || this.referenceHeight)?.toFixed(2)} cm
                    </div>
                    <div class="deviation-info">
                        Deviation: W±${detection.w_diff?.toFixed(2)}cm, H±${detection.h_diff?.toFixed(2)}cm
                    </div>
                    <div class="tolerance-info">
                        Tolerance: ±${currentTolerance?.toFixed(2)}cm ${detection.is_defect ? '(EXCEEDED)' : '(WITHIN LIMITS)'}
                    </div>
                </div>
            `;
            
            logContainer.appendChild(logEntry);
        });
    }

    animateNumber(elementId, targetValue, suffix = '') {
        const element = document.getElementById(elementId);
        if (!element) {
            console.warn(`Element ${elementId} not found for animation`);
            return;
        }
        
        // Get current value, handling edge cases
        let currentText = element.textContent || '0';
        if (suffix) {
            currentText = currentText.replace(suffix, '');
        }
        const currentValue = parseFloat(currentText) || 0;
        
        // If values are the same, no need to animate
        if (currentValue === targetValue) {
            return;
        }
        
        console.log(`Animating ${elementId}: ${currentValue} → ${targetValue}${suffix}`);
        
        const increment = (targetValue - currentValue) / 10;
        let current = currentValue;
        let steps = 0;
        const maxSteps = 10;
        
        const timer = setInterval(() => {
            current += increment;
            steps++;
            
            if (steps >= maxSteps || 
                (increment > 0 && current >= targetValue) || 
                (increment < 0 && current <= targetValue)) {
                current = targetValue;
                clearInterval(timer);
            }
            
            // Update display
            if (suffix === '%') {
                element.textContent = current.toFixed(1) + suffix;
            } else {
                element.textContent = Math.round(current) + suffix;
            }
        }, 50);
    }

    updateQualityIndicator(qualityRate) {
        const indicator = document.getElementById('qualityIndicator');
        if (!indicator) return;
        
        indicator.className = 'quality-indicator';
        
        if (qualityRate >= 95) {
            indicator.classList.add('excellent');
        } else if (qualityRate >= 90) {
            indicator.classList.add('good');
        } else if (qualityRate >= 80) {
            indicator.classList.add('warning');
        } else {
            indicator.classList.add('critical');
        }
    }

    showCalibrationModal() {
        const modal = document.getElementById('calibrationModal');
        if (modal) {
            // Set current values
            document.getElementById('referenceWidth').value = this.referenceWidth;
            document.getElementById('referenceHeight').value = this.referenceHeight;
            document.getElementById('toleranceValue').value = this.tolerance;
            
            modal.style.display = 'flex';
        }
    }

    hideCalibrationModal() {
        const modal = document.getElementById('calibrationModal');
        if (modal) {
            modal.style.display = 'none';
        }
    }

    async saveCalibration() {
        const width = parseFloat(document.getElementById('referenceWidth').value);
        const height = parseFloat(document.getElementById('referenceHeight').value);
        const tolerance = parseFloat(document.getElementById('toleranceValue').value);
        
        if (isNaN(width) || isNaN(height) || isNaN(tolerance) || 
            width <= 0 || height <= 0 || tolerance <= 0) {
            alert('Please enter valid positive numbers');
            return;
        }
        
        try {
            const response = await fetch(`${this.baseURL}/calibrate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    width: width,
                    height: height,
                    tolerance: tolerance
                })
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.referenceWidth = width;
                this.referenceHeight = height;
                this.tolerance = tolerance;
                
                this.showSystemStatus(`Calibrated: ${width}×${height}cm ±${tolerance}cm`, 'success');
                this.hideCalibrationModal();
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            console.error('Calibration error:', error);
            alert('Calibration failed: ' + error.message);
        }
    }

    async saveSettings() {
        const minArea = parseInt(document.getElementById('minArea').value);
        const maxArea = parseInt(document.getElementById('maxArea').value);
        
        try {
            const response = await fetch(`${this.baseURL}/settings`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    min_area_pixels: minArea,
                    max_area_pixels: maxArea,
                    reference_object_w: this.referenceWidth,
                    reference_object_h: this.referenceHeight,
                    tolerance: this.tolerance
                })
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.showSystemStatus('Settings saved successfully', 'success');
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            console.error('Settings error:', error);
            this.showSystemStatus('Failed to save settings: ' + error.message, 'error');
        }
    }

    resetStatistics() {
        if (confirm('Are you sure you want to reset all statistics?')) {
            // This would need a backend endpoint to actually reset
            this.showSystemStatus('Statistics reset (restart required)', 'warning');
        }
    }

    updateControlButtons() {
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        
        if (startBtn && stopBtn) {
            if (this.isRunning) {
                startBtn.disabled = true;
                stopBtn.disabled = false;
                startBtn.classList.add('disabled');
                stopBtn.classList.remove('disabled');
            } else {
                startBtn.disabled = false;
                stopBtn.disabled = true;
                startBtn.classList.remove('disabled');
                stopBtn.classList.add('disabled');
            }
        }
    }

    showSystemStatus(message, type = 'info') {
        const statusElement = document.getElementById('systemStatus');
        if (statusElement) {
            statusElement.textContent = message;
            statusElement.className = `system-status ${type}`;
            
            // Auto-clear success/error messages
            if (type === 'success' || type === 'error') {
                setTimeout(() => {
                    if (statusElement.textContent === message) {
                        statusElement.textContent = this.isRunning ? 'Accurate inspection active' : 'Ready for accurate inspection';
                        statusElement.className = `system-status ${this.isRunning ? 'success' : 'info'}`;
                    }
                }, 3000);
            }
        }
    }

    updateDisplay() {
        this.updateControlButtons();
        
        // Show current calibration
        const calibrationInfo = document.getElementById('calibrationInfo');
        if (calibrationInfo) {
            calibrationInfo.textContent = `Reference: ${this.referenceWidth}×${this.referenceHeight}cm ±${this.tolerance}cm`;
        }
    }

    playNotificationSound(type) {
        // Create audio context for notification sounds
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            if (type === 'start') {
                oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
                oscillator.frequency.setValueAtTime(1000, audioContext.currentTime + 0.1);
            } else if (type === 'stop') {
                oscillator.frequency.setValueAtTime(600, audioContext.currentTime);
                oscillator.frequency.setValueAtTime(400, audioContext.currentTime + 0.1);
            }
            
            gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);
            
            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.2);
        } catch (error) {
            console.log('Audio notification not supported');
        }
    }
}

// Initialize the accurate system when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.accurateOptiCount = new AccurateOptiCount();
    
    // Add some visual feedback
    document.body.classList.add('system-loaded');
});
