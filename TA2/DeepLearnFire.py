import cv2
from ultralytics import YOLO

# --- MODEL AND CAMERA CONFIGURATION ---

# Load the optimized TensorRT model
# For the first run, it might take a moment to build the engine
model_path = 'best.engine' 

# If you want to use the .pt file directly (slower), uncomment below
# model_path = 'best.pt' 

# URL from your cam_server.py on your laptop
# Make sure your laptop and Jetson are on the same WiFi network
# Replace with your laptop's IP address
url = "http://192.168.1.12:5000/video"

# Or use a USB camera connected to the Jetson Nano
# url = 0 
# --- END CONFIGURATION ---


# Load the YOLOv8 model
model = YOLO(model_path)

# Open the video source
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print(f"Error: Could not open video stream at {url}")
    exit()

print("Successfully connected to video stream. Starting detection...")

while True:
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        # The 'stream=True' argument is efficient for video processing
        results = model(frame, stream=True)

        # Process results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get confidence score and class name
                confidence = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])
                class_name = model.names[cls]

                # Draw bounding box and label on the frame
                if confidence > 0.5: # Only show detections with > 50% confidence
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    label = f'{class_name.upper()} {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.putText(frame, "FIRE DETECTED", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


        # Display the annotated frame
        cv2.imshow("YOLOv8 Fire Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the stream ends
        print("Video stream ended.")
        break

# Release the video capture object and close display windows
cap.release()
cv2.destroyAllWindows()