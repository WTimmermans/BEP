import cv2
import numpy as np
from cameradetect import detect_cameras
import platform

def nothing(x):
    pass

# === Camera Selection ===
print("Scanning for Cameras....")
camera_list = detect_cameras()

if not camera_list:
    print("No cameras found.")
    exit()

print("\nAvailable Cameras:")
for idx, (i, name) in enumerate(camera_list):
    print(f"{idx}: {name}")
    
selected_idx = int(input("Select a camera from the list:"))
camera_index = camera_list[selected_idx][0]

# === Start Calibration ===
# Open camera
if platform.system() == 'Windows':
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # For Windows, try DirectShow
else:
    cap = cv2.VideoCapture(camera_index)  # Default for macOS/Linux

# Create a window
cv2.namedWindow("Calibration")

# Create trackbars for colour change
cv2.createTrackbar("H Lower", "Calibration", 0, 179, nothing)
cv2.createTrackbar("S Lower", "Calibration", 0, 255, nothing)
cv2.createTrackbar("V Lower", "Calibration", 0, 255, nothing)

cv2.createTrackbar("H Upper", "Calibration", 179, 179, nothing)
cv2.createTrackbar("S Upper", "Calibration", 255, 255, nothing)
cv2.createTrackbar("V Upper", "Calibration", 255, 255, nothing)

print("\nUse sliders to adjust HSV thresholds.")
print("Press 's' to save values, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current positions of all trackbars
    hL = cv2.getTrackbarPos("H Lower", "Calibration")
    sL = cv2.getTrackbarPos("S Lower", "Calibration")
    vL = cv2.getTrackbarPos("V Lower", "Calibration")
    hU = cv2.getTrackbarPos("H Upper", "Calibration")
    sU = cv2.getTrackbarPos("S Upper", "Calibration")
    vU = cv2.getTrackbarPos("V Upper", "Calibration")

    lower = np.array([hL, sL, vL])
    upper = np.array([hU, sU, vU])

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        color_name = input("Enter color name to save (e.g., orange): ").strip()
        print(f"{color_name}_lower = {lower.tolist()}")
        print(f"{color_name}_upper = {upper.tolist()}")
        
cap.release()
cv2.destroyAllWindows()

