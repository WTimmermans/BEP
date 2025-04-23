import cv2
import numpy as np
import platform
import tkinter as tk
from tkinter import messagebox
from cameradetect import detect_cameras

# Load colour values
colourtable = np.load("colourvalues.npz")

def make_mask(frame, colour):
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(HSV, colourtable[colour + "_lower"], colourtable[colour + "_upper"])
    moments = cv2.moments(mask)

    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = 0, 0

    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    cv2.putText(frame, f"{colour}: ({cx},{cy})", (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return cx, cy, mask, frame

def start_camera():
    selection = camera_listbox.curselection()
    if not selection:
        messagebox.showerror("Error", "Please select a camera.")
        return

    selected_index = selection[0]
    cam_index = cameras[selected_index][0]

    #CAP_DSHOW only works in windows, so skip if on mac or linux
    if platform.system() == 'Windows':
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # For Windows, try DirectShow
    else:
        cap = cv2.VideoCapture(0)  # Default for macOS/Linux

    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open camera.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        _, _, _, frame = make_mask(frame, "orange")
        _, _, _, frame = make_mask(frame, "green")

        cv2.imshow("Live Webcam Feed, press q to close.", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Tkinter setup
root = tk.Tk()
root.title("Camera Selector")

tk.Label(root, text="Select a camera from the list:").pack(pady=(10, 0))

# Get camera list
cameras = detect_cameras()  # List of (index, name)

camera_listbox = tk.Listbox(root, height=6, width=50)
for i, (index, name) in enumerate(cameras):
    camera_listbox.insert(tk.END, f"[{index}] {name}")
camera_listbox.pack(padx=10, pady=10)
start_button = tk.Button(root, text="Start Camera", command=start_camera)
start_button.pack(pady=20)

root.mainloop()
