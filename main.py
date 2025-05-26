"""
=== MAIN ===

This is the main script for the tracking of circular stickers with the goal 
of measuring (vertical) distance change. This then corresponds to the 
deflection of a beam. From there the bending moment and shear force are 
calculated.

Adjust parameters within 'circles' to match markers and (lighting) situation.

Created by Steffen Scheelings and Wouter Timmermans for BEP at TU Delft 2025
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
from scipy.optimize import curve_fit


# Storage for deflection tracking
locked_positions = []  # Empty variable to store locked positions.
deflections = []
scale = 1
calibration_text = ""
calibration_counter = 0
known_distance_mm = 500
beam_list = ["Square aluminium", "C profile aluminium", "Hollow round steel", "Solid round steel", "Solid round POM"]

# MAYBE AT 'U'SECTION (90 DEG ROTATED C SECTION)

# Beam Properties (Use elif statements to add more profiles)
def beam_props(beam_select):
    # Square Aluminium
    if beam_select == 0:
        E = 69e9   # Young's Modulus Aluminium (Pa)
        b = 0.01   # Outer length (m)
        t = 0.001  # Thickness (m)
        I = ((b**4) - (b - 2*t)**4) / 12  # Second moment of interia (m^4)
        EI = E*I

    
    # C Profile Aluminium
    elif beam_select == 1:
        E = 69e9 # Young's Modulus Aluminium (Pa)
        t = 1e-3  # thickness (m)
        h = 10e-3  # height (m)
        b = 10e-3  # width (m)
        I = ((1/12)*t*(h-2*t)**3)+2*(((1/12)*b*t**3)+((b*t)*(((h/2)-(t/2))**2)))
        EI = E*I

        
    # Hollow Round Steel    
    elif beam_select == 2:
        E = 200e9 # Young's Modulus Steel (Pa)
        R = 6e-3 # External radius (m)
        r = 5e-3 # Internal radius (m)
        I = (np.pi/4)*(R**4-r**4) #m^2
        EI = E*I

    # Solid Round Steel
    elif beam_select == 3:
        E = 200e9 # Young's Modulus Steel (Pa)
        R = 4e-3 # External radius (m)
        I = (np.pi/4)*R**4 #m^4
        EI = E*I

    # Solid Round POM
    elif beam_select == 4:
        E = 2700e6 # Young's Modulus POM (Pa)
        R = 5e-3 # External radius (m)
        I = (np.pi/4)*R**4 #m^4
        EI = E*I
    
    else:
        print("Error: Please select beam.")

    return EI

# Shared key state
key_state = {
    'space_pressed': False,
    'q_pressed': False,
    'c_pressed': True
}
key_lock = Lock()

# Update variables
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

# Actual process that listens for input keys
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
        param1=250,     # Upper threshold for Canny edge detector (Circle contrast)
        param2=18,      # Threshold for center detection (Circle "perfectness")
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

# Calibrated distance using the centres of two outer circles 
def calibrate(circles, known_distance_mm):
    global scale, calibration_text, calibration_counter

    # Check that there are at least 2 circles
    if len(circles) < 2:
        calibration_text = "Need at least 2 circles to calibrate!"
        calibration_counter = 40
        print(calibration_text)
        return
    
    # Select first and last circle 
    circle0 = circles[0]
    circleN = circles[-1]

    # Calculate distances between first and last circle
    dx = abs(circleN[0] - circle0[0])
    dy = abs(circleN[1] - circle0[1])

    print(f"dx: {dx}, dy: {dy}")
    print(f"circle0: {circle0}, circleN: {circleN}")
    
    # Calculate the distance (Pythagoras)
    pixel_dist = np.sqrt(dx**2 + dy**2)

    # This handles NaN errors
    if any(np.isnan(val) for val in [*circle0[:2], *circleN[:2]]):
        calibration_text = "Calibration failed, invalid circle coordinates!"
        calibration_counter = 40
        print(calibration_text)
        return
 
    # This handles miscalculation errors
    if pixel_dist == 0:
        calibration_text = "Calibration failed, zero pixel distance detected!"
        calibration_counter = 40
        print(calibration_text)
        return
    
    # Caclulate mm/pixel scale
    scale = known_distance_mm /pixel_dist
    calibration_text = f"Calibration complete: {pixel_dist:.2f} pixels = {known_distance_mm} mm, scale = {scale:.4f} mm/pixel"
    calibration_counter = 40
    print(calibration_text)

# Main function: Initialises camera. Circle detection and colour detection.
def start_camera():
    global calibration_counter
    global deflections
    
    camera_selection = camera_listbox.curselection()
    beam_selection = beam_listbox.curselection()

    if not camera_selection or not beam_selection:
        messagebox.showerror("Error", "Please select a camera and a beam.")
        return

    #geeft de camera en beam select als 1 nummertje
    beam_select = beam_selection[0]
    selected_index = camera_selection[0]
    cam_index = cameras[selected_index][0]

    print(beam_props(beam_select))

    #CAP_DSHOW only works in windows, so skip if on mac or linux
    if platform.system() == 'Windows':
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)  # For Windows, try DirectShow
    else:
        cap = cv2.VideoCapture(cam_index)  # Default for macOS/Linux
    
    #Resolutie buiten de if statement geplaatst voor netheid
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
    
    # Second figure for moment and shear
    fig2, (ax_moment, ax_shear) = plt.subplots(2, 1, figsize=(8,6))
    plt.show(block=False)
    moment_theory_plot, = ax_moment.plot([], [], 'r', label='Theory Moment')
    shear_theory_plot, = ax_shear.plot([], [], 'r', label='Theory Shear')

    for axx in [ax_moment, ax_shear]:
        axx.axhline(0, color='green', linestyle="--")
        axx.set_xlim(0, 100)
        axx.legend()
        
    ax_moment.set_title("Moment Diagram")
    ax_moment.set_ylabel("Moment (N.mm)")   
    
    ax_shear.set_title("Shear Diagram")
    ax_shear.set_ylabel("Shear (N)")    
    ax_shear.set_xlabel("Langth along beam (mm)")
    

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
        if scale is not None and locked_positions and len(circles) == len(locked_positions):
            deflections = [(curr[1] - ref[1]) * scale for curr, ref in zip(circles, locked_positions)] #in mm
            deflect_plot.set_data(range(len(deflections)), deflections)
            
            # Set plot axis size
            ax_deflect.set_xlim(0, len(deflections))
            ax_deflect.set_ylim(50, -50)
        
            if deflections and len(deflections) >= 3:
                try:
                    EI = beam_props(beam_select)
                    xs_real = [x * scale for x in xs]
                    x_mm = np.array(xs_real)
                    y_mm = np.array(deflections)

                    # Convert to meters
                    x_m = x_mm / 1000.0
                    y_m = y_mm / 1000.0
                    L = np.max(x_m)

                    # Bending Moment and Shear Force Diagram
                    def theoretical_deflection(x, P):
                        return (P / (6 * EI)) * (3 * L * x**2 - x**3)

                    P_opt, _ = curve_fit(lambda x, P: theoretical_deflection(x, P), x_m, y_m)
                    P_fit = P_opt[0]

                    M_theory = -P_fit * (L - x_m)
                    V_theory = np.full_like(x_m, -P_fit)

                    #print(M_theory, V_theory)

                    # ==== Plotting ====
                    moment_theory_plot.set_data(x_mm, M_theory)
                    shear_theory_plot.set_data(x_mm, V_theory)

                    ax_moment.set_xlim(min(x_mm), max(x_mm))
                    ax_shear.set_xlim(min(x_mm), max(x_mm))

                    ax_moment.set_ylim(-25, 25)
                    ax_shear.set_ylim(-40, 40)

                    ax_moment.legend()
                    ax_shear.legend()

                except Exception as e:
                    print(f"Error in moment/shear calculation: {e}")
            
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        fig2.canvas.draw()
        fig2.canvas.flush_events()

        #on screen text for the camera feed goes here:
        if calibration_counter > 0:
            cv2.putText(frame, calibration_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            calibration_counter -= 1

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
                calibrate(circles, known_distance_mm)

            if key_state['q_pressed']:
                break
  
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()  # Turn off interactive mode for plots

# Tkinter setup
root = tk.Tk()
root.title("Camera Selector")
tk.Label(root, text="Select a camera from the list:").pack(pady=(10, 0))
#beam_list

# Get camera list
cameras = detect_cameras()  # List of (index, name)
if platform.system() == "Darwin" and not shutil.which("ffmpeg"):    # Ensures camera works on Mac
    messagebox.showerror("Warning", "ffmpeg is not installed, please install ffmpeg to get camera names.")

# Display camera list in window.
camera_listbox = tk.Listbox(root, height=6, width=50, exportselection=False)
for i, (index, name) in enumerate(cameras):
    camera_listbox.insert(tk.END, f"[{index}] {name}")
camera_listbox.pack(padx=10, pady=10)

# Display beam list in window
tk.Label(root, text="Now select a beam:").pack()

beam_listbox = tk.Listbox(root, height=6, width=50, exportselection=False)
for beam in beam_list:
    beam_listbox.insert(tk.END, beam)
beam_listbox.pack(pady=(0, 10))

# start camera (camera initialisation and circle detection) starts when button is pressed.
start_button = tk.Button(root, text="Start Camera", command=start_camera)
start_button.pack(pady=20)

# Start key listener thread
listener_thread = Thread(target=key_listener, daemon=True)
listener_thread.start()

root.mainloop()