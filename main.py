import cv2
from cameradetect import detect_cameras

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
cap = cv2.VideoCapture(cameraindex)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Continuously read frames
while True:
    # Read a frame
    ret, frame = cap.read()

    # If frame read was not successful, break the loop
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv2.imshow('Live Webcam Feed', frame)

    # Press 'q' to quit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
