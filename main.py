"""
=== MAIN ===

This is the main script for the tracking of circular stickers with the goal 
of measuring (vertical) distance change. This then corresponds to the 
deflection of a beam. From there the bending moment and shear force are 
calculated.

Adjust parameters within 'HOUGH_CIRCLES_PARAMS' to match markers and (lighting) situation.

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

# --- Configuration Constants ---
KNOWN_DISTANCE_MM = 500.0   # Known distance for calibration in mm
FRAME_CROP_Y_START = 300    # Crop frame to eliminate ambient noise
FRAME_CROP_Y_END = 570
FRAME_CROP_X_START = 0
FRAME_CROP_X_END = 1080

# --- HoughCircles parameters ---
HOUGH_CIRCLES_PARAMS = {
    'dp': 1.2,         # Inverse ratio of resolution
    'minDist': 50,     # Minimum distance between detected centres
    'param1': 300,     # Upper threshold for Canny edge detector (Circle contrast)
    'param2': 23,      # Threshold for center detection (Circle "perfectness")
    'minRadius': 1,    # Minimum circle radius
    'maxRadius': 10    # Maximum circle radius
}

# --- Beam Properties Data ---
# Structure: {"Name": {"E": Young's Modulus (Pa), "I_func": lambda R, r, b, h, t: I (m^4), "params": {dims}}}
# List of dictionaries. Uses less memory than other method and is easier to expand.
# Lambda function is just a def but inline. Leads to less lines.
BEAM_PROPERTIES = [
    {
        "name": "Square Aluminium",
        "E": 69e9,
        "I_func": lambda b, t, **kwargs: ((b**4) - (b - 2*t)**4) / 12,
        "params": {"b": 0.01, "t": 0.001} # b=Outer length, t=Thickness
    },
    {
        # U-profile opening leftwards or rightwards
        "name": "C Profile Aluminium", # U-profile when measuring deflection in y-dir as per image
        "E": 69e9,
        "I_func": lambda b, h, t, **kwargs: ((t * (h - 2*t)**3) / 12) + 2 * (((b * t**3) / 12) + (b * t) * (((h - t) / 2)**2)),
        "params": {"b": 10e-3, "h": 10e-3, "t": 1e-3} # b=flange width, h=height, t=thickness
    },
    {
         "name": "U Profile Aluminium ", # U profile opening upwards (or downwards)
         "E": 69e9,
         "I_func": lambda b, h, t, **kwargs: (1 / 6) * t * h**3 + ((1 /12)*(b - 2 * t) * t**3 + (b - 2 * t) * t * ((h / 2)-(t / 2))**2),
         "params": {"b": 10e-3, "h": 10e-3, "t": 1e-3} # b=flange width, h=height, t=thickness
    },
    {
        "name": "Hollow Round Steel",
        "E": 200e9,
        "I_func": lambda R, r, **kwargs: (np.pi/4)*(R**4-r**4),
        "params": {"R": 6e-3, "r": 5e-3} # R=External radius, r=Internal radius
    },
    {
        "name": "Solid round steel",
        "E": 200e9,
        "I_func": lambda R, **kwargs: (np.pi/4)*R**4,
        "params": {"R": 4e-3} # R=Radius
    },
    {
        "name": "Solid Round POM",
        "E": 2.7e9,
        "I_func": lambda R, **kwargs: (np.pi/4)*R**4,
        "params": {"R": 5e-3} # R=Radius
    }
]

BEAM_LIST_NAMES = [beam["name"] for beam in BEAM_PROPERTIES]

# --- Global State Variables (Shared across threads/callbacks) ---
locked_positions = []  # Stores locked (reference) circle positions (x, y) in pixels
deflections_mm = []    # Stores calculated deflections (current_y - locked_y) * scale in mm
pixel_scale = 1.0      # placeholder mm per pixel, determined during calibration
calibration_info = {"text": "", "counter": 0, "active": False} # For displaying calibration status

# Shared key state for keyboard listener
key_state = {
    'space_pressed': False,
    'q_pressed': False,
    'c_pressed': False
}

key_lock = Lock()

# --- Beam Properties Calculation ---
def get_beam_EI(beam_index):
    """Calculates EI (Flexural Rigidity) for the selected beam."""
    if 0 <= beam_index < len(BEAM_PROPERTIES):
        beam = BEAM_PROPERTIES[beam_index]
        E = beam["E"]
        I = beam["I_func"](**beam["params"])
        return E * I
    else:
        messagebox.showerror("Error", "Invalid beam selection.")
        return None

# --- Keyboard Input Handling ---
def on_press(key):
    """Handles key press events."""
    global key_state, key_lock
    try:
        with key_lock:
            if key == keyboard.Key.space:
                key_state['space_pressed'] = True
            elif hasattr(key, 'char'):
                if key.char == 'q':
                    key_state['q_pressed'] = True
                elif key.char == 'c':
                    key_state['c_pressed'] = True
    except AttributeError:
        pass # Ignore other keys not handled

def key_listener_thread():
    """Listens for keyboard input in a separate thread."""
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

# --- Image Processing ---
def detect_circles_in_frame(frame):
    """Detects circles in a given frame."""
    # Grayscale and blur the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    # Perform Circle Detection
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, **HOUGH_CIRCLES_PARAMS)

    output_circles = []
    if circles is not None and len(circles[0]) >= 2:
        # Filter out NaN values and convert to float32
        valid_circles = [c for c in circles[0, :] if not (np.isnan(c[0]) or np.isnan(c[1]))]
        processed_circles = np.around(np.array(valid_circles, dtype=np.float32)).astype(np.float32)

        if len(processed_circles) >= 2:
            # Sort circles by x-coordinate for consistent indexing
            processed_circles = sorted(processed_circles, key=lambda c: c[0])
            
            for i, (x, y, r) in enumerate(processed_circles):
                output_circles.append((x, y, r)) # (x_centre, y_centre, radius)
                
                # Draw detected circles and centers on the frame
                cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2)     # Green circle
                cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), 3)          # Red centre
                cv2.putText(frame, f"#{i} ({int(x)},{int(y)})", (int(x), int(y) + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)      # Blue text
    return output_circles

# --- Calibration ---
def perform_calibration(detected_circles):
    """Calibrates the pixel_scale using the two outermost detected circles.
       These are placed a known distance apart."""
    global pixel_scale, calibration_info

    if len(detected_circles) < 2:
        calibration_info.update({"text": "Calibration failed: Need at least 2 circles!", "counter": 60})
        print(calibration_info["text"])
        return False

    circle0 = detected_circles[0]   # First circle (leftmost due to sorting)
    circleN = detected_circles[-1]  # Last circle (rightmost due to sorting)

    dx = abs(circleN[0] - circle0[0])
    dy = abs(circleN[1] - circle0[1])
    pixel_dist = np.sqrt(dx**2 + dy**2)

    if pixel_dist == 0:
        calibration_info.update({"text": "Calibration failed: Zero pixel distance!", "counter": 60})
        print(calibration_info["text"])
        return False
    
    pixel_scale = KNOWN_DISTANCE_MM / pixel_dist
    calibration_info.update({
        "text": f"Calibrated: {pixel_dist:.2f}px = {KNOWN_DISTANCE_MM}mm. Scale: {pixel_scale:.4f} mm/px",
        "counter": 120 # Display for longer
    })
    print(calibration_info["text"])
    return True

# --- Theoretical Deflection Function (Cantilever Beam with Point Load at position 'a') ---
def general_cantilever_deflection_theory(x_positions_m, load_N, load_pos_a_m, EI_Nm2):
    """
    Calculates theoretical deflection for a cantilever beam with a point load P
    at a distance 'a' from the fixed end. This is a piecewise function.

    v(x) = (P * x^2) / (6 * EI) * (3*a - x)  for 0 <= x <= a
    v(x) = (P * a^2) / (6 * EI) * (3*x - a)  for a < x
    """
    x = np.asarray(x_positions_m)
    a = load_pos_a_m
    P = load_N
    EI = EI_Nm2

    # Define the conditions for the two parts of the beam (before and after the load)
    condlist = [x <= a, x > a]
    
    # Define the deflection equations for each respective condition
    funclist = [
        lambda x_val: (P * x_val**2 * (3*a - x_val)) / (6 * EI),
        lambda x_val: (P * a**2 * (3*x_val - a)) / (6 * EI)
    ]
    
    return np.piecewise(x, condlist, funclist)

# --- Main Camera and Processing Function ---
def start_camera_processing():
    """Initializes camera, performs detection, calculation, and plotting."""
    # --- Initialise Camera ---
    global locked_positions, deflections_mm, pixel_scale, calibration_info, key_state

    with key_lock:
        key_state["q_pressed"] = False # Reset flag

    camera_selection = camera_listbox.curselection()
    beam_selection_idx = beam_listbox.curselection()

    if not camera_selection or not beam_selection_idx:
        messagebox.showerror("Error", "Please select a camera and a beam.")
        return

    selected_beam_idx = beam_selection_idx[0]
    selected_cam_idx_in_list = camera_selection[0]
    
    # Get the actual camera index from the detected camera list
    if not (0 <= selected_cam_idx_in_list < len(available_cameras)):
        messagebox.showerror("Error", "Invalid camera selection from list.")
        return
    cam_hw_index = available_cameras[selected_cam_idx_in_list][0]

    EI_current_beam = get_beam_EI(selected_beam_idx)
    if EI_current_beam is None:
        return # Error already shown in get_beam_EI

    print(f"Selected Beam: {BEAM_LIST_NAMES[selected_beam_idx]}, EI: {EI_current_beam:.2f} Nm^2")

    cap_api = cv2.CAP_DSHOW if platform.system() == 'Windows' else cv2.CAP_ANY
    cap = cv2.VideoCapture(cam_hw_index, cap_api)
    # Try for higher resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Check actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Attempted resolution 1080x720. Actual: {actual_width}x{actual_height}")

    if not cap.isOpened():
        messagebox.showerror("Error", f"Cannot open camera index {cam_hw_index}.")
        return    
    
    # --- Plotting Setup ---
    plt.ion()
    # Setup for circle position plot
    fig_positions, (ax_positions, ax_deflection) = plt.subplots(2, 1, figsize=(8, 7))
    live_scatter = ax_positions.scatter([], [], label='Live Markers')
    locked_scatter = ax_positions.scatter([], [], c='red', marker='x', label='Locked Markers')
    live_line, = ax_positions.plot([], [], 'b-', lw=1)
    locked_line, = ax_positions.plot([], [], 'r-', lw=1)
    
    ax_positions.set_xlabel("X Position (pixels)")
    ax_positions.set_ylabel("Y Position (pixels)")
    ax_positions.set_title("Marker Positions (Y vs X)")
    ax_positions.invert_yaxis()
    ax_positions.legend()
    ax_positions.grid(True)

    # Setup for deflection plot
    # Measured deflections will be magenta circles and line
    deflection_plot, = ax_deflection.plot([], [], 'mo-', label='Deflection (ΔY Measured)') 
    # ADDED: Line for fitted deflection data (green dashed line)
    fitted_deflection_plot, = ax_deflection.plot([], [], 'g--', label='Deflection (Fitted Theory)')
    ax_deflection.set_title("Vertical Deflection per Marker")
    # CHANGED: X-axis label to reflect physical distance
    ax_deflection.set_xlabel("Position along beam (mm)")
    ax_deflection.set_ylabel("ΔY (mm)")
    ax_deflection.axhline(0, color='gray', linestyle='--', lw=1)
    ax_deflection.legend() # Will now include both labels
    ax_deflection.grid(True)
    fig_positions.tight_layout()

    # Setup for moment and shear plot
    fig_beam_analysis, (ax_moment, ax_shear) = plt.subplots(2, 1, figsize=(8, 7))
    moment_fit_plot, = ax_moment.plot([], [], 'g-', label='Bending Moment')
    shear_fit_plot, = ax_shear.plot([], [], 'c-', label='Shear Force')
    
    # Add gray 'zero-line'
    for ax_bm in [ax_moment, ax_shear]:
        ax_bm.axhline(0, color='gray', linestyle='--')
        ax_bm.legend()
        ax_bm.grid(True)
        
    ax_moment.set_title("Bending Moment Diagram")
    ax_moment.set_ylabel("Moment (N·m)") # Using Nm
    ax_shear.set_title("Shear Force Diagram")
    ax_shear.set_ylabel("Shear Force (N)")    
    ax_shear.set_xlabel("Position along beam (mm)")
    fig_beam_analysis.tight_layout()
    plt.show(block=False)

    # --- Main Processing Loop ---
    # To store current iteration's calculated x_coords_mm and deflections_mm for plotting
    current_x_coords_mm_for_plot = np.array([])
    current_deflections_mm_for_plot = np.array([])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Crop frame to region of interest
        cropped_frame = frame[FRAME_CROP_Y_START:FRAME_CROP_Y_END, FRAME_CROP_X_START:FRAME_CROP_X_END]
        
        current_circles = detect_circles_in_frame(cropped_frame) # circles are (x,y,r) in cropped_frame coordinates

        # --- Handle Keyboard Input ---
        with key_lock:
            if key_state['q_pressed']:
                break
            if key_state['c_pressed']:
                key_state['c_pressed'] = False # Reset flag
                calibration_info["active"] = True # Activate calibration mode
                print("Calibration mode activated. Ensure markers are in reference position.")
            if key_state['space_pressed']:
                key_state['space_pressed'] = False
                if current_circles:
                    locked_positions = [(c[0], c[1]) for c in current_circles]
                    print(f"Locked {len(locked_positions)} positions: {locked_positions}")
                    # Clear previous deflections
                    deflections_mm.clear() 
                    current_deflections_mm_for_plot = np.array([]) # Clear plot data
                    current_x_coords_mm_for_plot = np.array([])  # Clear plot data
                    ax_deflection.set_ylim(-10, 10) # Reset y-limit for deflection
                else:
                    print("Space pressed, but no circles detected to lock.")

        # --- Perform Calibration if Activated ---
        if calibration_info["active"]:
            perform_calibration(current_circles)
            calibration_info["active"] = False # Deactivate after one attempt

        # --- Update Live Plot Data (Positions) ---
        if current_circles:
            xs = [c[0] for c in current_circles]
            ys = [c[1] for c in current_circles]
            live_scatter.set_offsets(np.c_[xs, ys])
            live_line.set_data(xs, ys)
            ax_positions.set_xlim(0, cropped_frame.shape[1])
            ax_positions.set_ylim(cropped_frame.shape[0], 0) # Invert y-axis to match OpenCV coordinates
        else:
            live_scatter.set_offsets(np.empty((0,2)))
            live_line.set_data([], [])

        if locked_positions:
            lx, ly = zip(*locked_positions)
            locked_scatter.set_offsets(np.column_stack((lx, ly)))
            locked_line.set_data(lx, ly)
        else:
            locked_scatter.set_offsets(np.empty((0, 2)))
            locked_line.set_data([],[])
            
        # --- Calculate and Plot Deflections ---
        # MODIFIED: To use x_coords_mm for the x-axis of the deflection plot
        # and to dynamically adjust y-limits.
        if pixel_scale != 1.0 and locked_positions and len(current_circles) == len(locked_positions):
            # Calculate deflections and store in the global list
            deflections_mm.clear()
            for curr, ref in zip(current_circles, locked_positions):
                deflections_mm.append((curr[1] - ref[1]) * pixel_scale)
            
            current_deflections_mm_for_plot = np.array(deflections_mm) # For plotting this iteration

            if locked_positions: # Ensure locked_positions is not empty
                 x_coords_pixels_locked = np.array([pos[0] for pos in locked_positions])
                 if x_coords_pixels_locked.size > 0:
                    current_x_coords_mm_for_plot = (x_coords_pixels_locked - x_coords_pixels_locked[0]) * pixel_scale
                    deflection_plot.set_data(current_x_coords_mm_for_plot, current_deflections_mm_for_plot)
                    
                    # Dynamic X-limits for deflection plot
                    min_x_plot = np.min(current_x_coords_mm_for_plot) - 10 # 10mm padding
                    max_x_plot = np.max(current_x_coords_mm_for_plot) + 10
                    ax_deflection.set_xlim(min_x_plot, max_x_plot)

                    # Initial Y-limits based on measured data only (will be updated if fit occurs)
                    if current_deflections_mm_for_plot.size > 0:
                        min_y_measured = np.min(current_deflections_mm_for_plot)
                        max_y_measured = np.max(current_deflections_mm_for_plot)
                        padding = (max_y_measured - min_y_measured) * 0.1 + 5 # 10% or 5mm
                        ax_deflection.set_ylim(max_y_measured + padding, min_y_measured - padding) # Inverted Y
                    else:
                        ax_deflection.set_ylim(10, -10) # Default if no deflection data
                 else: # Not enough data for x-coordinates
                    deflection_plot.set_data([],[])
                    current_x_coords_mm_for_plot = np.array([])
                    ax_deflection.set_ylim(10, -10) 
            else: # No locked positions
                deflection_plot.set_data([],[])
                current_deflections_mm_for_plot = np.array([])
                current_x_coords_mm_for_plot = np.array([])
                ax_deflection.set_ylim(10, -10) 
        else: # Conditions for deflection calculation not met
            deflection_plot.set_data([],[])
            fitted_deflection_plot.set_data([], []) # Clear fitted plot too
            current_deflections_mm_for_plot = np.array([])
            current_x_coords_mm_for_plot = np.array([])
            ax_deflection.set_ylim(10, -10)

        # --- Beam Analysis (Moment and Shear) ---
        # Ensure we have enough data points (at least 4 for this derivative-based method)
        # and that calibration has been done (pixel_scale is not its default 1.0)
        if pixel_scale != 1.0 and locked_positions and len(deflections_mm) >= 4:
            try:
                # --- Stage 1: Detect Force Location using Derivative Analysis ---
                print("\n--- Beam Analysis Start ---")
                # Use the x_coords and y_coords calculated for the deflection plot
                # These are already relative to the first marker and in mm.
                x_coords_mm_analysis = current_x_coords_mm_for_plot 
                y_coords_mm_analysis = current_deflections_mm_for_plot

                print(f"X Coords (mm) for Analysis: {x_coords_mm_analysis}")
                print(f"Y Deflections (mm) for Analysis: {y_coords_mm_analysis}")

                if len(x_coords_mm_analysis) < 2:
                    raise ValueError("Not enough data points for derivative analysis (dx calculation).")

                dx = np.diff(x_coords_mm_analysis)
                if np.any(dx <= 0): # Check for non-positive differences as well
                    # Attempt to filter out non-increasing x-values if they cause issues
                    valid_indices = np.where(np.concatenate(([True], dx > 0)))[0]
                    if len(valid_indices) < 2:
                        raise ValueError("Too few valid x-coordinates after filtering non-increasing values for slope calculation.")
                    x_coords_mm_analysis = x_coords_mm_analysis[valid_indices]
                    y_coords_mm_analysis = y_coords_mm_analysis[valid_indices]
                    dx = np.diff(x_coords_mm_analysis)
                    if np.any(dx <= 0): # If still an issue
                         raise ValueError("Zero or negative difference in x-coordinates found; cannot calculate slopes reliably.")
                 
                if len(y_coords_mm_analysis) < 2: # Check after potential filtering
                     raise ValueError("Not enough data points for derivative analysis (dy calculation).")
                dy = np.diff(y_coords_mm_analysis)
                slopes = dy / dx
                print(f"Calculated slopes: {slopes}")

                SLOPE_STD_DEV_THRESHOLD = 0.005 # THIS ONE WE SHOULD CHANGE (PROLLY HIGHER)
                print(f"Slope STD DEV Threshold: {SLOPE_STD_DEV_THRESHOLD}")

                force_dot_index = len(x_coords_mm_analysis) - 1 # Index in the potentially filtered analysis arrays
                print(f"Initial force_dot_index (marker index for load 'a'): {force_dot_index}")

                if len(slopes) >= 2:
                    for i in range(len(slopes) - 2, -1, -1):
                        slopes_in_linear_region = slopes[i:]
                        current_std_dev = np.std(slopes_in_linear_region)
                        print(f"  Checking slopes from index {i} to end: {slopes_in_linear_region}")
                        print(f"  Std Dev of this region: {current_std_dev:.6f}")
                        
                        if current_std_dev > SLOPE_STD_DEV_THRESHOLD:
                            force_dot_index = i + 1
                            print(f"  Std Dev > Threshold. New force_dot_index: {force_dot_index} (marker {i+1} where load 'a' is applied)")
                            break
                        else:
                            print("  Std Dev <= Threshold. Region considered linear. Continuing search for kink further left.")
                    if not (current_std_dev > SLOPE_STD_DEV_THRESHOLD): 
                        print(f"  Loop completed. All tested slope sub-regions were linear or only one region tested. Final force_dot_index: {force_dot_index}")
                else:
                    print("  Not enough slopes (less than 2) to perform iterative std dev check. Using default force_dot_index.")

                print(f"Derivative Analysis Final: Force presumed at Marker index #{force_dot_index} (0-indexed in analysis array).")

                # --- Stage 2: Fit for Force Magnitude using the Detected Location ---
                x_m = x_coords_mm_analysis / 1000.0 # Convert to meters for theory function
                y_m = y_coords_mm_analysis / 1000.0 # Convert to meters for theory function
                
                if not (0 <= force_dot_index < len(x_m)): # Boundary check for force_dot_index
                    raise ValueError(f"Calculated force_dot_index {force_dot_index} is out of bounds for x_m array of length {len(x_m)}.")

                detected_pos_a_m = x_m[force_dot_index]
                print(f"Detected force position 'a': {detected_pos_a_m:.4f} m ({detected_pos_a_m * 1000:.1f} mm from fixed end)")
                
                def deflection_model_known_a(x_fit_data, P_fit):
                    return general_cantilever_deflection_theory(x_fit_data, P_fit, detected_pos_a_m, EI_current_beam)

                p0_guess_val = 1.0
                if len(y_m) > 0 and np.mean(y_m) < 0: # Check mean of deflections for guess direction
                    p0_guess_val = -1.0
                print(f"Initial guess for P_fit: {p0_guess_val}")

                popt, pcov = curve_fit(deflection_model_known_a, x_m, y_m, p0=[p0_guess_val])
                fitted_load_P = popt[0]

                print(f"Fitted Load P: {fitted_load_P:.2f} N")

                # --- Stage 2b: Calculate fitted deflection curve FOR PLOTTING ---
                # Use the x_coords_mm_analysis for plotting, which matches the measured data's x-axis.
                # The model expects x in meters, so we pass x_m (which is x_coords_mm_analysis/1000)
                y_fit_values_m = deflection_model_known_a(x_m, fitted_load_P)
                y_fit_values_mm = y_fit_values_m * 1000.0 # Convert back to mm for plotting

                fitted_deflection_plot.set_data(x_coords_mm_analysis, y_fit_values_mm)

                # Update Y-limits of ax_deflection to include both measured and fitted data
                if current_deflections_mm_for_plot.size > 0 and y_fit_values_mm.size > 0:
                    all_y_data = np.concatenate((current_deflections_mm_for_plot, y_fit_values_mm))
                    min_y_combined = np.min(all_y_data)
                    max_y_combined = np.max(all_y_data)
                    padding_y_combined = (max_y_combined - min_y_combined) * 0.1 + 5 # 5mm or 10%
                    ax_deflection.set_ylim(max_y_combined + padding_y_combined, min_y_combined - padding_y_combined) # Inverted
                elif y_fit_values_mm.size > 0: # Only fitted data available (less likely)
                    min_y_fit = np.min(y_fit_values_mm)
                    max_y_fit = np.max(y_fit_values_mm)
                    padding_y_fit = (max_y_fit - min_y_fit) * 0.1 + 5
                    ax_deflection.set_ylim(max_y_fit + padding_y_fit, min_y_fit - padding_y_fit) # Inverted


                # --- Stage 3: Calculate Moment and Shear ---
                moment_Nm_fit = np.piecewise(x_m, 
                                            [x_m <= detected_pos_a_m, x_m > detected_pos_a_m], 
                                            [lambda x_val: -fitted_load_P * (detected_pos_a_m - x_val), 
                                             0.0])
                
                shear_N_fit = np.piecewise(x_m, 
                                          [x_m < detected_pos_a_m, x_m >= detected_pos_a_m], 
                                          [-fitted_load_P, 
                                           0.0])

                moment_fit_plot.set_data(x_coords_mm_analysis, moment_Nm_fit)
                shear_fit_plot.set_data(x_coords_mm_analysis, shear_N_fit)

                ax_moment.set_ylim(min(moment_Nm_fit)*1.2 -1, max(moment_Nm_fit)*1.2 +1 ) # Dynamic
                ax_shear.set_ylim(min(shear_N_fit)*1.2 -1, max(shear_N_fit)*1.2 +1 )   # Dynamic
                
                if x_coords_mm_analysis.size > 0:
                    ax_moment.set_xlim(np.min(x_coords_mm_analysis) - 5, np.max(x_coords_mm_analysis) + 5)
                    ax_shear.set_xlim(np.min(x_coords_mm_analysis) - 5, np.max(x_coords_mm_analysis) + 5)
                print("--- Beam Analysis End ---")
                            
            except RuntimeError as e_curvefit:
                print(f"Curve fitting failed for beam analysis: {e_curvefit}")
                moment_fit_plot.set_data([],[])
                shear_fit_plot.set_data([],[])
                fitted_deflection_plot.set_data([], []) # Clear fitted deflection on error
                ax_moment.set_ylim(-1, 1)
                ax_shear.set_ylim(-1, 1)
                 # Reset y-lim for deflection plot if fit fails, to show only measured data
                if current_deflections_mm_for_plot.size > 0:
                    min_y_measured = np.min(current_deflections_mm_for_plot)
                    max_y_measured = np.max(current_deflections_mm_for_plot)
                    padding = (max_y_measured - min_y_measured) * 0.1 + 5
                    ax_deflection.set_ylim(max_y_measured + padding, min_y_measured - padding)
                else:
                    ax_deflection.set_ylim(10,-10)
                print("--- Beam Analysis Error (RuntimeError) ---")
            except ValueError as e_val:
                 print(f"Data validation error in beam analysis: {e_val}")
                 moment_fit_plot.set_data([],[])
                 shear_fit_plot.set_data([],[])
                 fitted_deflection_plot.set_data([], []) # Clear fitted deflection on error
                 ax_moment.set_ylim(-1, 1)
                 ax_shear.set_ylim(-1, 1)
                 if current_deflections_mm_for_plot.size > 0:
                    min_y_measured = np.min(current_deflections_mm_for_plot)
                    max_y_measured = np.max(current_deflections_mm_for_plot)
                    padding = (max_y_measured - min_y_measured) * 0.1 + 5
                    ax_deflection.set_ylim(max_y_measured + padding, min_y_measured - padding)
                 else:
                    ax_deflection.set_ylim(10,-10)
                 print("--- Beam Analysis Error (ValueError) ---")
            except Exception as e:
                print(f"Error in beam analysis: {e}")
                moment_fit_plot.set_data([],[])
                shear_fit_plot.set_data([],[])
                fitted_deflection_plot.set_data([], []) # Clear fitted deflection on error
                ax_moment.set_ylim(-1, 1)
                ax_shear.set_ylim(-1, 1)
                if current_deflections_mm_for_plot.size > 0:
                    min_y_measured = np.min(current_deflections_mm_for_plot)
                    max_y_measured = np.max(current_deflections_mm_for_plot)
                    padding = (max_y_measured - min_y_measured) * 0.1 + 5
                    ax_deflection.set_ylim(max_y_measured + padding, min_y_measured - padding)
                else:
                    ax_deflection.set_ylim(10,-10)
                print("--- Beam Analysis Error (Exception) ---")
        else: # Conditions for beam analysis not met
            moment_fit_plot.set_data([],[])
            shear_fit_plot.set_data([],[])
            fitted_deflection_plot.set_data([], []) # Clear fitted plot if analysis is skipped
            ax_moment.set_ylim(-1, 1)
            ax_shear.set_ylim(-1, 1)
            # Y-limits for ax_deflection would have been set by the "Calculate and Plot Deflections" section

        # --- Update Figure Canvases ---
        try:
            fig_positions.canvas.draw_idle()
            fig_positions.canvas.flush_events()
            fig_beam_analysis.canvas.draw_idle()
            fig_beam_analysis.canvas.flush_events()
        except Exception as e_plot:
            print(f"Error updating plots: {e_plot}")


        # --- Display Calibration Text on Frame ---
        if calibration_info["counter"] > 0:
            cv2.putText(cropped_frame, calibration_info["text"], (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            calibration_info["counter"] -= 1

        # Show Video Feed with Circle Detection
        cv2.imshow("Live Webcam Feed - Press 'q' to quit", cropped_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): # Fallback if pynput fails or for direct q
             break
  
    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    try:
        plt.close(fig_positions)
        plt.close(fig_beam_analysis)
    except Exception as e_close:
        print(f"Error closing plot figures: {e_close}")
    print("Camera and plots closed.")

# --- Tkinter GUI Setup ---
# (Rest of your Tkinter GUI setup code remains the same)
root = tk.Tk()
root.title("Beam Deflection Analyzer - Setup")

# Camera Selection
tk.Label(root, text="1. Select Camera:", font=('Helvetica', 10, 'bold')).pack(pady=(10,0), anchor='w', padx=10)
available_cameras = detect_cameras()
if platform.system() == "Darwin" and not shutil.which("ffmpeg"):
    messagebox.showwarning("ffmpeg Missing", "ffmpeg is not installed on this Mac. Camera names might not be available. Please install ffmpeg.")

camera_listbox = tk.Listbox(root, height=max(3, min(6, len(available_cameras))), width=60, exportselection=False)
if available_cameras:
    for i, (cam_idx, name) in enumerate(available_cameras):
        camera_listbox.insert(tk.END, f"[{cam_idx}] {name}")
else:
    camera_listbox.insert(tk.END, "No cameras detected.")
    camera_listbox.config(state=tk.DISABLED)
camera_listbox.pack(padx=10, pady=5, fill=tk.X)
if available_cameras: # Select first camera by default
    camera_listbox.select_set(0)

# Beam Selection
tk.Label(root, text="2. Select Beam Profile:", font=('Helvetica', 10, 'bold')).pack(pady=(10,0), anchor='w', padx=10)
beam_listbox = tk.Listbox(root, height=len(BEAM_LIST_NAMES), width=60, exportselection=False)
for beam_name in BEAM_LIST_NAMES:
    beam_listbox.insert(tk.END, beam_name)
beam_listbox.pack(padx=10, pady=5, fill=tk.X)
if BEAM_LIST_NAMES: # Select first beam by default
    beam_listbox.select_set(0)

# Instruction Text
tk.Label(root, text="Instructions:", font=('Helvetica', 10, 'bold')).pack(pady=(10,0), anchor='w', padx=10)
instructions_text = (
    "- Press 'c' to initiate calibration (ensure markers are at known distance).\n"
    "- Press 'SPACE' to lock current marker positions as reference.\n"
    "- Press 'q' in the feed window or GUI to quit."
)
tk.Label(root, text=instructions_text, justify=tk.LEFT).pack(padx=10, pady=5, anchor='w')

# "Start Analysis" Button Starts the Analysis Program
start_button = tk.Button(root, text="Start Analysis", command=start_camera_processing, font=('Helvetica', 12, 'bold'), bg='lightblue')
start_button.pack(pady=20, padx=10, fill=tk.X)

# Start keyboard listener thread (daemon so it exits with main)
kbd_listener_thread = Thread(target=key_listener_thread, daemon=True)
kbd_listener_thread.start()

root.mainloop()