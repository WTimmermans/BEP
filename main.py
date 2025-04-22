import cv2
from cameradetect import detect_cameras

if __name__ == "__main__":
    cameras = detect_cameras()
    if cameras:
        print("\nDetected Cameras:")
        for index, name in cameras:
            print(f"[{index}] {name}")
    else:
        print("No cameras detected.")


# Open a connection to the default webcam (0 is usually the default)

cap = cv2.VideoCapture(2)

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
