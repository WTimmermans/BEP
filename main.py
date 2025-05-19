"""
=== MAIN ===

This is the main script for the tracking of circular stickers with the goal 
of measuring (vertical) distance change. This then corresponds to the 
deflection of a beam.

Adjust parameters within 'circles' to match markers and situation.

Created by Steffen Scheelings and Wouter Timmermans 
For BEP at TU Delft 2025
"""

# Import relevant modules
import cv2
import numpy as np
import platform
import tkinter as tk
from tkinter import messagebox
from cameradetect import detect_cameras
import shutil
import matplotlib.pyplot as plt
from pynput import keyboard
from threading import Thread, Lock

# Shared key state
key_state = {
    'space_pressed': False,
    'q_pressed': False
}
key_lock = Lock()

# === This prevents bufferin of print statements in Windows.
# Might not be necessary ====
# import sys
# sys.stdout.reconfigure(line_buffering=True)


# Storage for deflection tracking
locked_positions = []  # Empty variable to store locked positions.

#update variables
def on_press(key):
    try:
        with key_lock:
            if key == keyboard.Key.space:
                key_state['space_pressed'] = True
            elif hasattr(key, 'char') and key.char == 'q':
                key_state['q_pressed'] = True
    except AttributeError:
        pass  # Handle special keys if needed

#actual process that listens for input keys
def key_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


# Detects circles within the camera feed
def detect_circle(frame):
    
    # Convert to grayscale and blur (to reduce noise)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    # Hough circle detect with adjustable paramaters
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
        dp=1.2,         # Inverse ratio of resolution
        minDist=50,     # Minimum distance between detected centres
        param1=70,      # Upper threshold for Canny edge detector
        param2=40,      # Threshold for center detection
        minRadius=1,    # Minimum circle radius
        maxRadius=20    # Maximum circle radius
    )

    output_circles = [] # Define output as an array

    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        
        # Sort circles left to right based on x for consistent indexing
        circles = sorted(circles, key=lambda c: c[0])
        
        for i, (x, y, r) in enumerate(circles):
            output_circles.append((x, y, r))  # (x_centre, y_centre, radius) 
            
            # Draw the circle green in the output image
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
            # Label the coordinates
            cv2.putText(frame, f"#{i}, ({x},{y})", (x+10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                  
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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    else:
        cap = cv2.VideoCapture(cam_index)  # Default for macOS/Linux
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open camera.")
        return    
    
    # Setup for live plotting
    plt.ion()  # Interactive mode on
    fig, ax = plt.subplots()
    plt.show(block=False)
    scatter = ax.scatter([], [], label='Live')
    locked_scatter = ax.scatter([], [], c='red', marker='x', label='Locked')
    line, = ax.plot([], [], 'b-', lw=1)  # 'b-' = blue line, lw=1 for line width for live
    line2, =ax.plot([],[], 'r-', lw=1)   # 'r-' = red line, lw=1 for line width for locked
    
    ax.set_xlabel("X Position (pixels)")
    ax.set_ylabel("Y Position (pixels)")
    ax.set_title("Live Marker Positions (Y vs X)")
    ax.invert_yaxis()   # Y increases downward in image coordinates
    ax.legend()         # Show legend   

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        #crop for unnessecary pixels
        frame = frame[0:720,100:880]

        # Call the circle detect funtion.
        circles = detect_circle(frame)

        # Live plot update
        if len(circles) > 0:
            xs = [c[0] for c in circles]
            ys = [c[1] for c in circles]

            scatter.set_offsets(np.c_[xs, ys])  # Update scatter plot
            line.set_data(xs, ys)              # Update line plot

            # Fix axis limits to avoid autoscale jumping
            ax.set_xlim(0, frame.shape[1])
            ax.set_ylim(frame.shape[0], 0)

            fig.canvas.draw()
            fig.canvas.flush_events()

        if locked_positions:
            lx, ly = zip(*locked_positions)
            locked_scatter.set_offsets(np.column_stack((lx, ly)))
            line2.set_data(lx, ly)
        else:
            locked_scatter.set_offsets(np.empty((0, 2)))

        # Show resulting image with circles marked.
        cv2.imshow("Live Webcam Feed, press q to close.", frame)
        cv2.waitKey(1)

        # Handle key presses from listener
        with key_lock:
            if key_state['space_pressed']:
                key_state['space_pressed'] = False
                print("Space Pressed")

                if circles:
                    print("Circles detected")
                    locked_positions.clear()
                    locked_positions.extend([(c[0], c[1]) for c in circles])
                    print("Locked positions:", locked_positions, flush=True)
                else:
                    print("No circles detected")

            if key_state['q_pressed']:
                break

        # Optional: still allow closing with window [X]
        if cv2.getWindowProperty("Live Webcam Feed, press q to close.", cv2.WND_PROP_VISIBLE) < 1:
            break
            
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()  # Turn off interactive mode for plots

# Tkinter setup
root = tk.Tk()
root.title("Camera Selector")
tk.Label(root, text="Select a camera from the list:").pack(pady=(10, 0))

# Get camera list
cameras = detect_cameras()  # List of (index, name)
if platform.system() == "Darwin" and not shutil.which("ffmpeg"):    # Ensures camera works on Mac
    messagebox.showerror("Warning", "ffmpeg is not installed, please install ffmpeg to get camera names.")

# Display camera list and selection window.
camera_listbox = tk.Listbox(root, height=6, width=50)
for i, (index, name) in enumerate(cameras):
    camera_listbox.insert(tk.END, f"[{index}] {name}")
camera_listbox.pack(padx=10, pady=10)

# start camera (camera initialisation and circle detection) starts when button is pressed.
start_button = tk.Button(root, text="Start Camera", command=start_camera)
start_button.pack(pady=20)

# Start key listener thread
listener_thread = Thread(target=key_listener, daemon=True)
listener_thread.start()

root.mainloop()