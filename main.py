import cv2
import numpy as np
from cameradetect import detect_cameras

#load colour values from file:
colourtable = np.load("colourvalues.npz")

def make_mask(frame, colour):
    HSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(HSV, colourtable[colour + "_lower"], colourtable[colour+"_upper"])
    moments = cv2.moments(mask)

    if moments["m00"] != 0:
        #calculate the centroid average of the pixels:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx = 0
        cy = 0

    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    cv2.putText(frame, f"{colour}: ({cx},{cy})", (cx + 10, cy - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return cx, cy, mask, frame

#make sure no weird stuff happens when trying to input an integer
def get_int(prompt):
    while True:
        try:
            value = int(input(prompt))
            return value
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

cameras = detect_cameras()

if cameras:
    print("\nDetected Cameras:")
    for index, name in cameras:
        print(f"[{index}] {name}")
else:
    print("No cameras detected.")

cameraindex = get_int("Please input the camera to use for the live feed: ")
print("Opening camera...")
cap = cv2.VideoCapture(cameraindex)

print("Opening camera window...")
# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Continuously read frames
while cap.isOpened():
    # Read a frame
    ret, frame = cap.read()
    #print(frame)
    cxorange, cyorange, orangemask, frame = make_mask(frame, "orange")
    cxgreen, cygreen, greenmask, frame = make_mask(frame, "green")



    # If frame read was not successful, break the loop
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv2.imshow('Live Webcam Feed, press q to close.', frame)
    #cv2.imshow("Mask",greenmask)

    # Press 'q' to quit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()