# Real-Time Object Tracking System (RTOS)

A computer vision project implementing intelligent traffic monitoring using YOLOv8 object detection with advanced state tracking and speed analysis. 

## 🎯 Project Motive

This project was undertaken to understand how traffic and tracking systems work in real time conditions and scenarios
- **Tracking multiple objects** simultaneously in traffic scenarios
- **Distinguishing between moving and stopped vehicles** to analyze traffic flow patterns
- **Eliminating false positives** (like stationary poles misidentified as people)
- **Calculating object speeds** for traffic monitoring
- **Maintaining temporal memory** to ensure consistent tracking across frames

The goal was to build a production-grade traffic monitoring solution that balances accuracy with real-time performance.

## 🏗️ Architecture & Key Design Decisions

### 1. **YOLOv8 Nano + Temporal Memory System**
- YOLOv8n model for optimal speed-accuracy tradeoff on consumer hardware
- Sophisticated `track_memory` maintains position history, timestamps, class voting (10-frame window), speed history (5-frame moving average), and state persistence
- Class voting mechanism ensures stable labels despite frame-to-frame detection variations

### 2. **State Machine with Data-Driven Thresholds**
Two-state system (Moving/Stopped) with histogram-analyzed thresholds:
```
STOPPED_THRESHOLD = 5 px/s    # From traffic speed distribution analysis
STATIC_THRESHOLD = 2 px/s     # Separates infrastructure from slow traffic
STOPPED_FRAME_COUNT = 10       # Hysteresis prevents oscillation
```
- Requires 10 consecutive frames below 5 px/s to transition to "Stopped"
- Objects < 2 px/s over 20+ frames classified as "ignore" (eliminates poles, signs)
- Visual feedback: Green (moving), Red (stopped)

### 3. **Speed Calculation & Performance Optimization**
- 10-frame window for speed calculation, 5-frame moving average for display
- Disabled MiDaS depth estimation for 2-3x FPS improvement
- Python dictionaries for O(1) track lookup, efficient numpy operations

## 🔧 Implementation Highlights

### Core Features
1. **Multi-Object Tracking**: Persistent ID assignment across frames using YOLO's built-in tracker
2. **Real-Time FPS Display**: Performance monitoring overlay
3. **Speed Histogram**: Statistical analysis with binned speed distribution (0-5, 5-10, 20-50, etc.)
4. **State Visualization**: Color-coded bounding boxes based on motion state
5. **Video Output**: Records annotated video with all tracking overlays
6. **Console Analytics**: Prints detailed statistics including min/max/mean/median speeds

### Technical Stack
- **Detection**: YOLOv8 (Ultralytics)
- **Computer Vision**: OpenCV (cv2)
- **Computation**: NumPy, PyTorch
- **Data Structures**: Python dictionaries for O(1) track lookup

### Performance Optimizations
- Disabled verbose logging (`verbose=False` in YOLO tracking)
- Uses nano model for faster inference
- Batch processing of frame data
- Efficient numpy operations for speed calculations

## 📊 Key Learnings & Takeaways

### 1. **Temporal Smoothing Prevents Instability**
- Simple thresholding creates jittery state transitions; hysteresis (10-frame requirement) essential for stable state machines
- Class voting across 10-frame history dramatically improves label consistency vs single-frame detection
- Motion-based filtering (< 2 px/s over 20 frames) reliably eliminates static object misclassifications

### 2. **Data-Driven Design Over Guesswork**
- Speed histogram analysis revealed clear separation between static objects (< 2 px/s) and stopped vehicles (< 5 px/s)
- Empirical threshold selection from traffic data distributions ensures robust performance across scenarios
- Statistical summaries (min/max/mean/median) essential for validation and parameter tuning

### 3. **Performance Tradeoffs in Real-Time Systems**
- Disabled depth estimation (MiDaS) for 2-3x FPS gain—prioritized core tracking over secondary features
- Real-time CV demands continuous profiling to balance accuracy vs latency
- Pixel-space metrics sufficient for relative analysis; coordinate transforms deferred until needed

## 🚀 Potential Future Enhancements
- Re-enable MiDaS with GPU acceleration for 3D traffic analysis
- Implement perspective transform for real-world speed calibration
- Add lane detection and traffic density heatmaps
- Integrate with actual traffic management systems
- Train custom YOLO model on specific traffic scenarios

## 📝 Technical Notes
- Requires YOLOv8n.pt model weights (included)
- Expects video input at `video_path` (configurable)
- Outputs annotated video as `output.mp4`
- Tested on 1080p traffic footage

---

**Project Focus**: Practical real-time computer vision with emphasis on robustness over novelty. Demonstrates production-ready engineering principles: performance optimization, state management, and false positive handling in real-world scenarios.
