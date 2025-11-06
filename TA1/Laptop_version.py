import cv2
from ultralytics import YOLO

# --- MODEL AND CAMERA CONFIGURATION ---

# Load the original PyTorch model you trained in Colab.
# Make sure 'best.pt' is in the same folder as this script.
model_path = 'best.pt' 

# Use '0' to access the default built-in laptop webcam.
url = 0 
# --- END CONFIGURATION ---


# Load the YOLOv8 model
# The task='detect' argument can help prevent warnings.
model = YOLO(model_path, task='detect')

# Open the video source
cap = cv2.VideoCapture("fire.mp4")

if not cap.isOpened():
    print(f"Error: Could not open video stream.")
    exit()

print("Successfully connected to webcam. Starting detection...")

while True:
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, stream=True)

        fire_detected_this_frame = False
        # Process results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                fire_detected_this_frame = True
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
        
        # Display a single "FIRE DETECTED" message if any fire was found in the frame
        if fire_detected_this_frame:
            cv2.putText(frame, "FIRE DETECTED", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Fire Detection (Laptop Demo)", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the stream ends
        print("Webcam stream ended.")
        break

# Release the video capture object and close display windows
cap.release()
cv2.destroyAllWindows()