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


# Storage for deflection tracking
locked_positions = []  # Empty variable to store locked positions.
deflections = []
scale = None
known_distance_mm = 500

# Shared key state
key_state = {
    'space_pressed': False,
    'q_pressed': False,
    'c_pressed': True
}
key_lock = Lock()

#update variables
def on_press(key):
    try:
        with key_lock:
            if key == keyboard.Key.space:
                key_state['space_pressed'] = True
            elif hasattr(key, 'char') and key.char == 'q':
                key_state['q_pressed'] = True
            elif hasattr(key, 'char') and key.char == 'c':
                key_state['c_pressed'] = True
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
        param1=300,     # Upper threshold for Canny edge detector (Circle contrast)
        param2=23,      # Threshold for center detection (Circle "perfectness")
        minRadius=1,    # Minimum circle radius
        maxRadius=10    # Maximum circle radius
    )

    output_circles = [] # Define output as an array

    if circles is not None and len(circles[0]) >= 2:
        circles = np.around(circles[0, :]).astype(np.float32)
        
        #filter our invalid values
        circles = [c for c in circles if not np.isnan(c[0]) and not np.isnan(c[1])]

        # Sort circles left to right based on x for consistent indexing
        circles = sorted(circles, key=lambda c: c[0])
        
        for i, (x, y, r) in enumerate(circles):
            output_circles.append((x, y, r))  # (x_centre, y_centre, radius) 
            
            # Draw the circle green in the output image
            cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), 3)
            # Label the coordinates
            cv2.putText(frame, f"#{i}, ({int(x)},{int(y)})", (int(x)+10, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)                  
                  
    return output_circles

def calibrate(circles, known_distance_mm):
    global scale

    if len(circles) < 2:
        print("Need at least 2 circles to calibrate!")
        return
    
    circle0 = circles[0]
    circleN = circles[-1]

    dx = abs(circleN[0] - circle0[0])
    dy = abs(circleN[1] - circle0[1])

    print(f"dx: {dx}, dy: {dy}")

    print(f"circle0: {circle0}, circleN: {circleN}")

    pixel_dist = np.sqrt(dx**2 + dy**2)

    if any(np.isnan(val) for val in [*circle0[:2], *circleN[:2]]):
        print("Invalid circle coordinates, calibration aborted.")
        return
 
    if pixel_dist == 0:
        print("Zero pixel distance detected!")
        return
    
    scale = known_distance_mm /pixel_dist
    print(f"Calibration complete: {pixel_dist:.2f} pixels = {known_distance_mm} mm → scale = {scale:.4f} mm/pixel")

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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080) # Set view window size
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        cap = cv2.VideoCapture(cam_index)  # Default for macOS/Linux
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open camera.")
        return    
    
    # Setup for live plotting
    plt.ion()  # Interactive mode on
    fig, (ax, ax_deflect) = plt.subplots(1, 2, figsize=(10,5))
    plt.show(block=False)
    scatter = ax.scatter([], [], label='Live')
    locked_scatter = ax.scatter([], [], c='red', marker='x', label='Locked')
    line, = ax.plot([], [], 'b-', lw=1)  # 'b-' = blue line, lw=1 for line width for live
    line2, =ax.plot([],[], 'r-', lw=1)   # 'r-' = red line, lw=1 for line width for locked
    deflect_plot = ax_deflect.plot([], [], 'ro-', label='ΔY (Deflection)')[0]
    ax_deflect.set_title("Vertical Deflection per Marker")
    ax_deflect.set_xlabel("Marker Index")
    ax_deflect.set_ylabel("ΔY (mm)")
    ax_deflect.axhline(0, color='gray', linestyle='--', lw=1)
    ax_deflect.legend()

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
        frame = frame[300:570,0:1080] # [upper:lower, left:right]

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
        
        # Plot locked positions and add line    
        if locked_positions:
            lx, ly = zip(*locked_positions)
            locked_scatter.set_offsets(np.column_stack((lx, ly)))
            line2.set_data(lx, ly)
        else:
            locked_scatter.set_offsets(np.empty((0, 2)))
        
      # Measure Difference between locked and currect vertical position
        if locked_positions and len(circles) == len(locked_positions):
            deflections = [curr[1] - ref[1] for curr, ref in zip(circles, locked_positions)] # calculate difference
            deflections_mm = [d/scale for d in deflections] # Scale delflection using calibration
            deflect_plot.set_data(range(len(deflections)), deflections) # Plot data
            
            # Set plot axis sizes
            ax_deflect.set_xlim(0, len(deflections_mm))
            ax_deflect.set_ylim(80, -80)

        # Measure Difference between locked and currect vertical position
        if scale is not None and locked_positions and len(circles) == len(locked_positions):
            deflections = [(curr[1] - ref[1]) * scale for curr, ref in zip(circles, locked_positions)] #in mm
            deflect_plot.set_data(range(len(deflections)), deflections)
            
            # Set plot axis sizes
            ax_deflect.set_xlim(0, len(deflections))
            ax_deflect.set_ylim(100, -100)


        fig.canvas.draw()
        fig.canvas.flush_events()
        # End Deflection Measure



        # Show resulting image with circles marked.
        cv2.imshow("Live Webcam Feed, press q to close.", frame)
        
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
                    
            if key_state['c_pressed']:
                key_state['c_pressed'] = False
                print("Calibration mode activated.")
                if circles:
                    calibrate(circles, known_distance_mm)
                else:
                    print("No circles detected to calibrate.")
            
            if key_state['q_pressed']:
                break

            if key_state['c_pressed']:
                key_state['c_pressed'] = False
                calibrate(circles, known_distance_mm)

            
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