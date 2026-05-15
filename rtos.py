import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import Counter

rtosmodel = YOLO("yolov8n.pt")  # Using nano model for speed

# MiDaS disabled for better performance
# midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
# device = torch.device("cpu")
# midas_model.to(device)
# midas_model.eval()

# Load MiDaS transforms
# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
# transform = midas_transforms.small_transform

# Open video file (change path to your video file)
video_path = r"C:\Users\ramprakashn\Downloads\vidssave.com 4K Road traffic video for object detection and tracking - free download now! 1080p.mp4"  # Change this to your video file path
cam = cv2.VideoCapture(video_path)

if not cam.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Temporal Memory or History Tracking
track_memory = {}

# Speed Tracker
all_speeds = []

# Histogram bins for speed distribution (in pixels per second)
bins = [0, 5, 10, 20, 50, 100, 200]
histogram = {}

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

prev_frame_time = 0
new_frame_time = 0

#Adjust alpha values to suit your need
alpha = 0.2
previous_depth = 0.0
    
#Applying exponential moving average filter to reduce variation in depth values across frames, which can help stabilize the depth estimation for detected objects. This is especially useful when the depth map is noisy or when there are rapid changes in depth due to camera movement or object motion.
def apply_ema_filter(current_depth):
    global previous_depth
    filtered_depth = alpha * current_depth + (1 - alpha) * previous_depth
    previous_depth = filtered_depth  # Update the previous depth value
    return filtered_depth


# Disabled MiDaS depth estimation as the project is more focussed on Traffic tracking and analysis
while True:
    ret, frame = cam.read()
    
    # Check if video ended
    if not ret:
        print("End of video or cannot read frame")
        break

    # Convert to RGB for MiDaS
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply MiDaS transform to resize, normalize, and convert to tensor and move to device
    # Done to match input requirements of the MiDaS model
    # input_batch = transform(img).to(device)

    # Run MiDaS model with no gradient calculation since we're only doing inference
    # To get Depth map for the current frame
    # with torch.no_grad():
    #     depth = midas_model(input_batch)

    # Remove batch dimension and convert to numpy array for visualization
    # depth_map = depth.squeeze().cpu().numpy()
    
    # print("Depth map:", depth_map.shape)

    # Normalize depth map for visualization
    # depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    # depth_colormap = cv2.applyColorMap(depth_normalized.astype('uint8'), cv2.COLORMAP_MAGMA)

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Run YOLO tracking on the frame
    results = rtosmodel.track(frame, persist=True, verbose=False)
    
    # Visualize the results on the frame
    annotated_frame = frame.copy()

    # Get tracked object information
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get class ID and name
            class_id = int(box.cls.cpu().numpy()[0])
            class_name = results[0].names[class_id]
            
            # Get confidence score
            confidence = float(box.conf.cpu().numpy()[0])

            # Get tracking ID (if available)
            if box.id is not None:
                track_id = int(box.id.cpu().numpy()[0])
                
                # Initialize tracking memory for new objects
                if track_id not in track_memory:
                    track_memory[track_id] = {
                        "positions": [],
                        "timestamps": [],
                        "class_history": [],
                        "speed_history": [],
                        "stable_class": None,
                        "last_seen": None,
                        "stopped_frames": 0,
                        "state": "Unknown"  # Start with unknown state
                    }
                
                # Calculate center point
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Update tracking memory
                track_memory[track_id]["positions"].append((cx, cy))
                track_memory[track_id]["timestamps"].append(new_frame_time)
                track_memory[track_id]["class_history"].append(class_name)
                track_memory[track_id]["last_seen"] = new_frame_time

                # Calculate the most common class in the history for this track to determine a stable class label
                if track_memory[track_id]["class_history"]:
                    recent_classes = track_memory[track_id]["class_history"][-10:]
                    stable_class = Counter(recent_classes).most_common(1)[0][0]
                else:
                    stable_class = "Unknown"
                
                track_memory[track_id]["stable_class"] = stable_class

                # Calculate speed if we have at least 2 positions
                # Calculate speed if we have enough data
                if len(track_memory[track_id]["positions"]) >= 10 and len(track_memory[track_id]["timestamps"]) >= 10:
                    # Get last two positions and timestamps
                    pos1 = track_memory[track_id]["positions"][-10]
                    pos2 = track_memory[track_id]["positions"][-1]
                    time1 = track_memory[track_id]["timestamps"][-10]
                    time2 = track_memory[track_id]["timestamps"][-1]
                    
                    # Calculate pixel distance
                    pixel_distance = ((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)**0.5
                    time_diff = time2 - time1
                    
                    if time_diff > 0:
                        speed_pixels_per_sec = pixel_distance / time_diff
                        track_memory[track_id]["speed_history"].append(speed_pixels_per_sec)
                        all_speeds.append(speed_pixels_per_sec)

                # Calculate average speed for display
                if track_memory[track_id]["speed_history"]:
                    avg_speed = np.mean(track_memory[track_id]["speed_history"][-5:])
                else:
                    avg_speed = 0

                # Eliminate false positives like poles being detected as persons
                if track_memory[track_id]["speed_history"]:
                    if avg_speed < 2 and len(track_memory[track_id]["speed_history"]) > 20:
                        print(f"Ignored items count for ID {track_id}: {len(track_memory[track_id]['speed_history'])} | Avg Speed: {avg_speed:.2f} px/s")
                        stable_class = "ignore"
                        track_memory[track_id]["stable_class"] = stable_class
                
                # Define speed threshold for stopped state
                STOPPED_THRESHOLD = 5  # pixels per second
                STOPPED_FRAME_COUNT = 10  # consecutive frames to consider stopped
                
                # State transition logic
                # MOVE -> STOPPED if speed is below threshold for enough frames
                # STOPPED -> MOVE if speed goes above threshold
                if avg_speed < STOPPED_THRESHOLD:
                    track_memory[track_id]["stopped_frames"] += 1
                    
                    # Transition to stopped if stopped for enough frames
                    if track_memory[track_id]["stopped_frames"] >= STOPPED_FRAME_COUNT:
                        if track_memory[track_id]["state"] != "Stopped":
                            print(f"ID {track_id} ({stable_class}) transitioned to STOPPED")
                        track_memory[track_id]["state"] = "Stopped"
                else:
                    # Moving - reset stopped counter
                    if track_memory[track_id]["stopped_frames"] > 0:
                        # Was stopped, now moving
                        if track_memory[track_id]["state"] == "Stopped":
                            print(f"ID {track_id} ({stable_class}) transitioned to MOVING (speed: {avg_speed:.1f} px/s)")
                        track_memory[track_id]["stopped_frames"] = 0
                    
                    track_memory[track_id]["state"] = "Moving"

                # Set display color based on state
                current_state = track_memory[track_id]["state"] or "Unknown"
                state_color = (0, 0, 255) if current_state == "Stopped" else (0, 255, 0)  # Red if stopped, green if moving

                if not (stable_class == "ignore"):
                    # Draw bounding box with state-based color
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), state_color, 2)
                    cv2.putText(annotated_frame, stable_class, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2)

                    # Add ID, speed, and state text to each object
                    cv2.putText(
                        annotated_frame,
                        f"ID:{track_id} {current_state} {avg_speed:.1f}px/s",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        state_color,
                        2
                    )

            else:
                track_id = -1  # No ID assigned yet

    # Add FPS text to the frame
    cv2.putText(annotated_frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the annotated frame to the output file
    out.write(annotated_frame)

    # Display the annotated frame and depth map
    cv2.imshow('Video - YOLO Tracking', annotated_frame)

    # cv2.imshow('Depth Map', depth_colormap)

    # Press 'q' to exit, or adjust delay (25ms ≈ 40fps playback)
    if cv2.waitKey(25) == ord('q'):
        break

    if len(all_speeds) > 100:

        print("\n========== SPEED STATS ==========")

        print(f"Samples : {len(all_speeds)}")
        print(f"Min     : {np.min(all_speeds):.2f}")
        print(f"Max     : {np.max(all_speeds):.2f}")
        print(f"Mean    : {np.mean(all_speeds):.2f}")
        print(f"Median  : {np.median(all_speeds):.2f}")

        print("=================================\n")

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()

# Print formatted track history
print("\n" + "="*80)
print(f"Frame Time: {new_frame_time:.2f} | Active Tracks: {len(track_memory)}")
print("="*80)

for i in range(len(bins) - 1):

    low = bins[i]
    high = bins[i + 1]

    count = sum(low <= s < high for s in all_speeds)

    histogram[f"{low}-{high}"] = count

print("\n========== SPEED HISTOGRAM ==========")

for k, v in histogram.items():
    print(f"{k:10} : {v}")

print("=====================================\n")

for track_id, data in track_memory.items():
    if not data["positions"]:
        continue
        
    # Calculate trajectory length
    trajectory_points = len(data["positions"])
    
    # Smooth speed using moving average of last 5 values
    if data["speed_history"]:
        smooth_speed = np.mean(data["speed_history"][-5:])
    else:
        smooth_speed = 0
        
    speed_display = f"{smooth_speed:.1f} px/s" if smooth_speed > 0 else "N/A"

    # Time since last seen
    time_since_seen = new_frame_time - data["last_seen"] if data["last_seen"] else 0

    # print(f"  ID {track_id:3d} | Class: {data['stable_class']:10s} | "
    #     f"Trajectory: {trajectory_points:3d} points | Speed: {speed_display:12s} | "
    #         f"Last pos: {data['positions'][-1] if data['positions'] else 'N/A'}")
    
print("="*80 + "\n")