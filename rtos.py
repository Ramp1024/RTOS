import time

import cv2
import torch
from ultralytics import YOLO

rtosmodel = YOLO("yolo26n.pt")
midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cpu")
midas_model.to(device)
midas_model.eval()

# Load MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

while True:
    ret, frame = cam.read()

    # Convert to RGB for MiDaS
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply MiDaS transform to resize, normalize, and convert to tensor and move to device
    # Done to match input requirements of the MiDaS model
    input_batch = transform(img).to(device)

    # Run MiDaS model with no gradient calculation since we're only doing inference
    # To get Depth map for the current frame
    with torch.no_grad():
        depth = midas_model(input_batch)

    # Remove batch dimension and convert to numpy array for visualization
    depth_map = depth.squeeze().cpu().numpy()
    
    print("Depth map:", depth_map.shape)

    # Normalize depth map for visualization
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_colormap = cv2.applyColorMap(depth_normalized.astype('uint8'), cv2.COLORMAP_MAGMA)

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Run YOLO tracking on the frame
    results = rtosmodel.track(frame, persist=True)
    
    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Add depth info to detected objects
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        print(f"Detected {len(results[0].boxes)} objects")  # Debug
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Ensure coordinates are within bounds
            cy = min(max(0, cy), depth_map.shape[0] - 1)
            cx = min(max(0, cx), depth_map.shape[1] - 1)

            depth_value = depth_map[cy, cx]
            filtered_depth = apply_ema_filter(depth_value)
            print(f"Object at ({cx}, {cy}), Depth: {filtered_depth:.2f}")  # Debug
            cv2.putText(annotated_frame, f'D: {filtered_depth:.1f}m', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Add FPS text to the frame
    cv2.putText(annotated_frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the annotated frame to the output file
    out.write(annotated_frame)

    # Display the annotated frame and depth map
    cv2.imshow('Camera - YOLO Tracking', annotated_frame)
    cv2.imshow('Depth Map', depth_colormap)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()