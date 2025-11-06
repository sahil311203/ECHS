import cv2
from ultralytics import YOLO
import time
import paho.mqtt.client as mqtt
import json
import uuid

# --- MODEL & CAMERA CONFIGURATION ---
# Use the new model you trained on Colab and converted to .engine
#MODEL_PATH = 'best.engine' 
# Or use the .pt file directly (slower)
MODEL_PATH = 'best.pt' 

#VIDEO_SOURCE https://pypi.ngc.nvidia.comVIDEO_SOURCE = "http://192.168.1.12:5000/video"
# Or use a USB camera
VIDEO_SOURCE = 0 

# --- AI AGENT LOGIC CONFIGURATION ---
# Detections must be above this confidence to be considered
CONFIDENCE_THRESHOLD = 0.5 
# A fire must be detected for this many consecutive frames to trigger an alert
FIRE_PERSISTENCE_FRAMES = 10 
# Wait this many seconds before sending another alert (prevents spam)
ALERT_COOLDOWN_SECONDS = 300 # 5 minutes

# --- MQTT CONFIGURATION ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "forest/fire/alert" # This MUST match what you set in your phone app
JETSON_ID = "JetsonOrin-Forest-Cam-01" # Give your device a unique name

# --- AGENT STATE (Internal variables) ---
fire_frame_counter = 0
last_alert_time = 0
is_alert_active = False

def mqtt_on_connect(client, userdata, flags, rc, properties=None):
    """Callback for when the client connects to the broker."""
    if rc == 0:
        print(f"Connected to MQTT Broker at {MQTT_BROKER}")
    else:
        print(f"Failed to connect, return code {rc}")

def publish_alert(client, class_name, confidence):
    """Publishes a JSON message to the MQTT broker."""
    global last_alert_time, is_alert_active
    
    current_time = time.time()
    # Check if we are in cooldown
    if (current_time - last_alert_time) > ALERT_COOLDOWN_SECONDS:
        print(f"--- !!! ALARM TRIGGERED !!! ---")
        print(f"Persistent {class_name} detected with {confidence:.2f} confidence.")
        
        # Create a JSON payload
        payload = {
            "device_id": JETSON_ID,
            "event_type": class_name.upper(),
            "confidence": f"{confidence:.2f}",
            "timestamp": int(current_time),
            "message": f"Fire alarm from {JETSON_ID}!"
        }
        
        # Publish the message
        result = client.publish(MQTT_TOPIC, json.dumps(payload))
        
        # Check if publish was successful
        if result[0] == 0:
            print(f"Successfully published alert to topic: {MQTT_TOPIC}")
        else:
            print(f"Failed to publish alert. Code: {result[0]}")
            
        # Update agent state
        last_alert_time = current_time
        is_alert_active = True
        
        # TODO: This is where you would start saving a video clip to your SSD
        # e.g., save_video_clip_to_ssd(frame_buffer)

# --- SETUP ---
# Load the YOLOv8 model
print(f"Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
class_names = model.names
print(f"Model loaded. Classes: {class_names}")

# Open the video source
print(f"Connecting to video stream: {VIDEO_SOURCE}...")
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open video stream.")
    exit()
print("Successfully connected to video stream.")

# Setup MQTT client


# Generate a unique client ID to prevent connection conflicts
client_id = f"jetson-agent-{uuid.uuid4()}" 
print(f"Connecting with unique Client ID: {client_id}")

# Use the new API version (fixes DeprecationWarning) and set the unique ID
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)

# --- (The rest of your script stays the same) ---
client.on_connect = mqtt_on_connect
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

print("AI Fire Agent is active. Starting detection loop...")

# --- MAIN LOOP ---
try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Video stream ended or frame failed.")
            # Optional: try to reconnect
            cap.release()
            time.sleep(5)
            cap = cv2.VideoCapture(VIDEO_SOURCE)
            if not cap.isOpened():
                print("Failed to reconnect. Exiting.")
                break
            continue

        # 1. PERCEIVE
        # Run YOLOv8 inference
        results = model(frame, stream=True, verbose=False)

        fire_detected_this_frame = False
        
        # 2. REASON
        for r in results:
            boxes = r.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = class_names[cls_id]

                # --- Check for FIRE ---
                if class_name == 'fire' and confidence > CONFIDENCE_THRESHOLD:
                    fire_detected_this_frame = True
                    
                    # Add persistence
                    fire_frame_counter += 1
                    
                    # --- REASONING -> ACTION ---
                    if fire_frame_counter >= FIRE_PERSISTENCE_FRAMES:
                        # We have a persistent fire, trigger the alert
                        publish_alert(client, class_name, confidence)
                    
                    # Draw box (for local debugging window)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    label = f'{class_name.upper()} {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    
                    # We found the fire, no need to check other boxes
                    break 
            
            # --- Check for SMOKER (different logic) ---
            # You could add separate logic for 'smoker' here if needed
            # e.g., log it or send a lower-priority alert
            
            if fire_detected_this_frame:
                break # Exit inner loop

        # --- Update Agent State ---
        if not fire_detected_this_frame:
            # If no fire is seen, reset the counter
            fire_frame_counter = 0
            if is_alert_active:
                print("Fire event concluded.")
                is_alert_active = False # Reset alarm state

        # Display the annotated frame (for debugging)
        if is_alert_active:
             cv2.putText(frame, "!!! FIRE ALARM ACTIVE !!!", (50, 50),
                         cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        cv2.imshow("YOLOv8 Fire Agent", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Shutting down agent...")
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    client.loop_stop()
    client.disconnect()
    print("Agent stopped.")