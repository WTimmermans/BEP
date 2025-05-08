### This is the main script for the tracking of coloured circular stickers
### with the goal of measuring distance change.

### When lisitn gcolour related variables the following order is (always) used:
### yellow, lightgreen, darkgreen, aquamarine. turquoise, lightblue, darkblue,
### purple, violet, red, orange, pink, brown.
### This colour sequence is defined by the sticker sheet.

### Created by Steffen Scheelings and Wouter Timmermans 
### For BEP at TU Delft 2025

# ====  BROWN IS CURRENTLY 'TURNED OFF', TO TURN ON -> ADD TO 'COLOURS' VARIABLE===


# Import relevant modules
import cv2
import numpy as np
import platform
import tkinter as tk
from tkinter import messagebox
from cameradetect import detect_cameras
import shutil


# Load colour values
colourtable = np.load("colourvalues.npz")

# Define EMPTY centroid location variables
locked_centroids = {
    "yellow": None,
    "lightgreen": None,
    "darkgreen": None,
    "aquamarine": None,
    "turquoise": None,
    "lightblue": None,
    "darkblue": None,
    "purple": None,
    "violet": None,
    "red": None,
    "orange": None,
    "pink": None,
    #"brown": None
    
}


# Creates a mask (negative image) of a colour
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

# Detects circles within the camera feed
def detect_circle(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.medianBlur(gray, 5)
    
    # Hough circle detect with adjustable paramaters
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

# Main function: Initialises camera. Circle detection and colour detection.
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

        circles = detect_circle(frame)

        if circles:
            for (x, y, r) in circles:
                circle_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.circle(circle_mask, (x, y), r, 128, thickness=-1)

                associated = False
                
                colours = ["yellow", "lightgreen", "darkgreen", "aquamarine",
                           "turquoise", "lightblue", "darkblue", "purple",
                           "violet", "red", "orange", "pink"]

                for colour in colours:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)        # Predefine empty mask
                    cx, cy, mask = make_mask(frame, colour, extra_mask=circle_mask) # Make coloured mask
                    if cx !=0 and cy !=0:
                        associated = True
                        if locked_centroids[colour] is not None:
                            lx, ly = locked_centroids[colour]
                            cv2.line(frame, (lx, ly), (cx, cy), (255, 0, 0), 2)
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                        cv2.putText(frame, f"{colour}: ({cx},{cy})", (cx + 10, cy-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                if associated:
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 2)  # Only draw the (green) circle if associated
                else:
                        cv2.circle(frame, (x, y), r, (0, 0, 255), 2)  # Red circle if unassociated
                        cv2.putText(frame, "Unassociated", (x - 20, y - r - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.imshow("Live Webcam Feed, press q to close.", frame)
        # cv2.imshow("Live Green Mask Feed, press q to close.", lightgreenmask)     I guess these are useless now.
        # cv2.imshow("Live Orange Mask Feed, press q to close.", orangemask)
        
        # Quit programm by pressing 'q' on keyboard or [X] on screen.
        key = cv2.waitKey(1) & 0xFF
        
        if cv2.getWindowProperty("Live Webcam Feed, press q to close.", cv2.WND_PROP_VISIBLE) < 1:
            break
        
        # Press q to quit program
        if key == ord('q'):
            break
        
        # Press space to lock centroids
        elif key == ord(' '):
            for colour in colours:
                cx, cy, _ = make_mask(frame, colour)
                if cx != 0 and cy != 0:
                    locked_centroids[colour] = (cx, cy)
                    print(f"Locked {colour} at {locked_centroids[colour]}")
        

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

# Display camera list and selection window.
camera_listbox = tk.Listbox(root, height=6, width=50)
for i, (index, name) in enumerate(cameras):
    camera_listbox.insert(tk.END, f"[{index}] {name}")
camera_listbox.pack(padx=10, pady=10)

# start camera (camera initialisation, circle & colour detection) starts when 
# button is pressed.
start_button = tk.Button(root, text="Start Camera", command=start_camera)
start_button.pack(pady=20)

root.mainloop()