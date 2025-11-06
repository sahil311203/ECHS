import cv2
import numpy as np
import pywt


# Function to compute wavelet energy features
def wavelet_energy(gray_img):
    # Apply 2D Discrete Wavelet Transform (Haar wavelet)
    coeffs2 = pywt.dwt2(gray_img, 'haar')
    LL, (LH, HL, HH) = coeffs2
    # Energy from high-frequency sub-bands
    energy = np.sum(np.square(LH)) + np.sum(np.square(HL)) + np.sum(np.square(HH))
    return energy

# Fire energy threshold (tuned experimentally from fire templates)
FIRE_ENERGY_THRESHOLD = 500000  


url = "http://192.168.123.29:5000/video"
cap = cv2.VideoCapture(url)
#cap = cv2.VideoCapture("/home/lyaiec/Desktop/Viresh D2233082/fireforest/fire2.mp4")  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Fire color range 
    lower = np.array([0, 50, 50])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological filters to reduce noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5,5), np.uint8))

    # Extract fire-like regions
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = cv2.bitwise_and(gray, gray, mask=mask)

    # Compute wavelet energy
    energy = wavelet_energy(roi)

    # Detect contours in mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fire_detected = False
    if energy > FIRE_ENERGY_THRESHOLD:
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:  # Ignore small areas
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                fire_detected = True

    if fire_detected:
        cv2.putText(frame, " FIRE DETECTED ", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Show outputs
    cv2.imshow("Forest Fire Detection", frame)
    #cv2.imshow("Fire Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()