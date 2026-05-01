import time

import cv2
from ultralytics import YOLO

rtosmodel = YOLO("yolo26n.pt")

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

while True:
    ret, frame = cam.read()

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Run YOLO tracking on the frame
    results = rtosmodel.track(frame, persist=True)
    
    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Add FPS text to the frame
    cv2.putText(annotated_frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the annotated frame to the output file
    out.write(annotated_frame)

    # Display the annotated frame
    cv2.imshow('Camera', annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()