import cv2
import numpy as np
import platform
import tkinter as tk
from tkinter import messagebox
from cameradetect import detect_cameras
import shutil


# Load colour values
colourtable = np.load("colourvalues.npz")

locked_centroids = {
    "orange": None,
    "green": None
}

def make_mask(frame, colour, extra_mask=None):
    global locked_centroids

    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(HSV, colourtable[colour + "_lower"], colourtable[colour + "_upper"])

    # Apply extra circle mask if given
    if extra_mask is not None:
        mask = cv2.bitwise_and(mask, mask, mask=extra_mask)

    moments = cv2.moments(mask)

    if moments["m00"]  > 2000:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = 0, 0

    # Draw a line from locked centroid to current
    if locked_centroids[colour] is not None:
        lx, ly = locked_centroids[colour]
        cv2.line(frame, (lx, ly), (cx, cy), (255, 0, 0), 2)

    # Mark current centroid
    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
    cv2.putText(frame, f"{colour}: ({cx},{cy})", (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return cx, cy, mask, frame

def detect_circle(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.medianBlur(gray, 5)
    
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=1,
        maxRadius=20
    )

    output_circles = []

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            output_circles.append((i[0], i[1], i[2]))  # (x, y, r)
            # Draw detected circles for visualization
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            #cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

    return frame, output_circles


def start_camera():
    selection = camera_listbox.curselection()
    if not selection:
        messagebox.showerror("Error", "Please select a camera.")
        return

    selected_index = selection[0]
    cam_index = cameras[selected_index][0]

    #CAP_DSHOW only works in windows, so skip if on mac or linux
    if platform.system() == 'Windows':
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)  # For Windows, try DirectShow
    else:
        cap = cv2.VideoCapture(cam_index)  # Default for macOS/Linux

    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open camera.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        frame, circles = detect_circle(frame)
    
        if circles:
            for (x, y, r) in circles:
                # Create a mask for just this circle
                circle_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.circle(circle_mask, (x, y), r, 255, thickness=-1)
    
                # Detect colors inside this circle on the full frame
                cx_orange, cy_orange, orangemask, frame = make_mask(frame, "orange", extra_mask=circle_mask)
                cx_green, cy_green, greenmask, frame = make_mask(frame, "green", extra_mask=circle_mask)
    
                # Optional: Draw center points
                cv2.circle(frame, (cx_orange, cy_orange), 3, (0, 165, 255), -1)  # Orange dot
                cv2.circle(frame, (cx_green, cy_green), 3, (0, 255, 0), -1)      # Green dot
    
        cv2.imshow("Live Webcam Feed, press q to close.", frame)
        cv2.imshow("Live Green Mask Feed, press q to close.", greenmask)
        cv2.imshow("Live Orange Mask Feed, press q to close.", orangemask)
    
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            locked_centroids["orange"] = (cx_orange, cy_orange)
            locked_centroids["green"] = (cx_green, cy_green)
            print(f"Locked orange at {locked_centroids['orange']}, green at {locked_centroids['green']}")
    


    cap.release()
    cv2.destroyAllWindows()

# Tkinter setup
root = tk.Tk()
root.title("Camera Selector")

tk.Label(root, text="Select a camera from the list:").pack(pady=(10, 0))

# Get camera list
cameras = detect_cameras()  # List of (index, name)
if platform.system() == "Darwin" and not shutil.which("ffmpeg"):
    messagebox.showerror("Warning", "ffmpeg is not installed, please install ffmpeg to get camera names.")

camera_listbox = tk.Listbox(root, height=6, width=50)
for i, (index, name) in enumerate(cameras):
    camera_listbox.insert(tk.END, f"[{index}] {name}")
camera_listbox.pack(padx=10, pady=10)
start_button = tk.Button(root, text="Start Camera", command=start_camera)
start_button.pack(pady=20)

root.mainloop()