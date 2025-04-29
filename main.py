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

    if extra_mask is not None:
        mask = cv2.bitwise_and(mask, mask, mask=extra_mask)

    moments = cv2.moments(mask)
    cx, cy = 0, 0

    if moments["m00"] > 2000:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

    return cx, cy, mask

def detect_circle(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.medianBlur(gray, 5)
    
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=70,
        param2=40,
        minRadius=5,
        maxRadius=40
    )

    output_circles = []

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            output_circles.append((i[0], i[1], i[2]))  # (x, y, r)
    return output_circles


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

        # Predefine empty masks
        greenmask = np.zeros(frame.shape[:2], dtype=np.uint8)
        orangemask = np.zeros(frame.shape[:2], dtype=np.uint8)

        circles = detect_circle(frame)

        if circles:
            for (x, y, r) in circles:
                circle_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.circle(circle_mask, (x, y), r, 128, thickness=-1)

                cx_orange, cy_orange, orangemask = make_mask(frame, "orange", extra_mask=circle_mask)
                cx_green, cy_green, greenmask = make_mask(frame, "green", extra_mask=circle_mask)

                associated = False

                if cx_orange != 0 and cy_orange != 0:
                    associated = True
                    if locked_centroids["orange"] is not None:
                        lx, ly = locked_centroids["orange"]
                        cv2.line(frame, (lx, ly), (cx_orange, cy_orange), (255, 0, 0), 2)
                    cv2.circle(frame, (cx_orange, cy_orange), 5, (255, 0, 0), -1)
                    cv2.putText(frame, f"orange: ({cx_orange},{cy_orange})", (cx_orange + 10, cy_orange - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                if cx_green != 0 and cy_green != 0:
                    associated = True
                    if locked_centroids["green"] is not None:
                        lx, ly = locked_centroids["green"]
                        cv2.line(frame, (lx, ly), (cx_green, cy_green), (255, 0, 0), 2)
                    cv2.circle(frame, (cx_green, cy_green), 5, (255, 0, 0), -1)
                    cv2.putText(frame, f"green: ({cx_green},{cy_green})", (cx_green + 10, cy_green - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                if associated:
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 2)  # Only draw the circle if associated

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