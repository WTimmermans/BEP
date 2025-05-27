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
KNOWN_DISTANCE_MM = 500.0  # Known distance for calibration in mm
FRAME_CROP_Y_START = 300
FRAME_CROP_Y_END = 570
FRAME_CROP_X_START = 0
FRAME_CROP_X_END = 1080

# HoughCircles parameters
HOUGH_CIRCLES_PARAMS = {
    'dp': 1.2,         # Inverse ratio of resolution
    'minDist': 50,     # Minimum distance between detected centres
    'param1': 250,     # Upper threshold for Canny edge detector (Circle contrast)
    'param2': 18,      # Threshold for center detection (Circle "perfectness")
    'minRadius': 1,    # Minimum circle radius
    'maxRadius': 10    # Maximum circle radius
}

# Beam Properties Data
# Structure: {"Name": {"E": Young's Modulus (Pa), "I_func": lambda R, r, b, h, t: I (m^4), "params": {dims}}}
#lijst van dictionaries, gebruikt minder geheugen dan de andere methode, en makkelijk aan toe te voegen (extra entry in de dictionary)
#je hoeft alleen de benodigde parameters in te voegen, de rest wordt genegeerd
BEAM_PROPERTIES = [
    {
        "name": "Square aluminium",
        "E": 69e9,
        #een lambda functie is gewoon een def maar dan inline, wat minder kut is voor je computer
        "I_func": lambda b, t, **kwargs: ((b**4) - (b - 2*t)**4) / 12,
        "params": {"b": 0.01, "t": 0.001} # b=Outer length, t=Thickness
    },
    {
        "name": "C profile asluminium", # U-profile when measuring deflection in y-dir as per image
        "E": 69e9,
        # For a C-channel bent about its strong axis (flanges vertical, web horizontal)
        # Assuming standard orientation where bending occurs around the x-axis (axis parallel to width b)
        # I = I_web + 2 * (I_flange_centroidal + A_flange * d_flange^2)
        # web: (t * (h-2t)^3) / 12
        # flange: (b * t^3) / 12
        # A_flange = b*t
        # d_flange = (h-t)/2
        "I_func": lambda b, h, t, **kwargs: ((t * (h - 2*t)**3) / 12) + 2 * (((b * t**3) / 12) + (b * t) * (((h - t) / 2)**2)),
        "params": {"b": 10e-3, "h": 10e-3, "t": 1e-3} # b=flange width, h=height, t=thickness
    },
    {
        "name": "Hollow round steel",
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
        "name": "Solid round POM",
        "E": 2.7e9, # Corrected from 2700e6 to 2.7e9 for consistency
        "I_func": lambda R, **kwargs: (np.pi/4)*R**4,
        "params": {"R": 5e-3} # R=Radius
    },
    #{
        #"name": "etc"
    #}
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
    #Checkt nu tegelijk of de selectie die je hebt gemaakt klopt en rekent dan de EI uit
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
        pass # Ignore special keys not handled

def key_listener_thread():
    #pleurt de key listener in een aparte thread
    """Listens for keyboard input in a separate thread."""
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

# --- Image Processing ---
def detect_circles_in_frame(frame):
    """Detects circles in a given frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
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
                cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2) # Green circle
                cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), 3)      # Red center
                cv2.putText(frame, f"#{i} ({int(x)},{int(y)})", (int(x) + 10, int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1) # Blue text
                            
    return output_circles

# --- Calibration ---
def perform_calibration(detected_circles):
    """Calibrates the pixel_scale using the two outermost detected circles."""
    global pixel_scale, calibration_info

    if len(detected_circles) < 2:
        calibration_info.update({"text": "Calibration failed: Need at least 2 circles!", "counter": 60})
        print(calibration_info["text"])
        return False

    circle0 = detected_circles[0]    # First circle (leftmost)
    circleN = detected_circles[-1]   # Last circle (rightmost)

    # Ensure coordinates are valid numbers
    #niet per se nodig, maar ook niet heel erg om erin te houden voor het geval de calibratie weer kut gaat doen
    if any(np.isnan(val) for val in [*circle0[:2], *circleN[:2]]):
        calibration_info.update({"text": "Calibration failed: Invalid circle coordinates!", "counter": 60})
        print(calibration_info["text"])
        return False

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

# --- Theoretical Deflection Function (Cantilever Beam with Point Load at End) ---
# PAS AAN VOOR ANDERE SITUATIES
def cantilever_deflection_theory(x_positions_m, load_N, EI_Nm2, length_L_m):
    """
    Calculates theoretical deflection for a cantilever beam with a point load P at the free end.
    v(x) = (P / (6 * EI)) * (3 * L * x^2 - x^3)
    where x is the distance from the fixed end.
    """
    # Ensure x is not greater than L to avoid issues with the formula's physical meaning
    x_clipped = np.clip(x_positions_m, 0, length_L_m)
    return (load_N / (6 * EI_Nm2)) * (3 * length_L_m * x_clipped**2 - x_clipped**3)

# --- Main Camera and Processing Function ---
def start_camera_processing():
    """Initializes camera, performs detection, calculation, and plotting."""
    global locked_positions, deflections_mm, pixel_scale, calibration_info, key_state

    with key_lock:
        key_state["q_pressed"]==False

    camera_selection = camera_listbox.curselection()
    beam_selection_idx = beam_listbox.curselection()

    if not camera_selection or not beam_selection_idx:
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
    selected_beam_idx = beam_selection_idx[0]
    selected_cam_idx_in_list = camera_selection[0]
    
    # Get the actual camera index from the detected cameras list
    if not (0 <= selected_cam_idx_in_list < len(available_cameras)):
        messagebox.showerror("Error", "Invalid camera selection from list.")
        return
    cam_hw_index = available_cameras[selected_cam_idx_in_list][0]


    EI_current_beam = get_beam_EI(selected_beam_idx)
    if EI_current_beam is None:
        return # Error already shown by get_beam_EI

    print(f"Selected Beam: {BEAM_LIST_NAMES[selected_beam_idx]}, EI: {EI_current_beam:.2f} Nm^2")

    cap_api = cv2.CAP_DSHOW if platform.system() == 'Windows' else cv2.CAP_ANY
    cap = cv2.VideoCapture(cam_hw_index, cap_api)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # Try for higher res
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Check actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Attempted resolution 1920x1080. Actual: {actual_width}x{actual_height}")


    if not cap.isOpened():
        messagebox.showerror("Error", f"Cannot open camera index {cam_hw_index}.")
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
    # --- Plotting Setup ---
    plt.ion()
    fig_positions, (ax_positions, ax_deflection) = plt.subplots(1, 2, figsize=(12, 6))
    live_scatter = ax_positions.scatter([], [], label='Live Markers')
    locked_scatter = ax_positions.scatter([], [], c='red', marker='x', label='Locked Markers')
    live_line, = ax_positions.plot([], [], 'b-', lw=1, label='Live Path')
    locked_line, = ax_positions.plot([], [], 'r-', lw=1, label='Locked Path')
    
    ax_positions.set_xlabel("X Position (pixels)")
    ax_positions.set_ylabel("Y Position (pixels)")
    ax_positions.set_title("Marker Positions (Y vs X)")
    ax_positions.invert_yaxis()
    ax_positions.legend()
    ax_positions.grid(True)

    deflection_plot, = ax_deflection.plot([], [], 'mo-', label='Deflection (ΔY)') # Magenta
    ax_deflection.set_title("Vertical Deflection per Marker")
    ax_deflection.set_xlabel("Marker Index")
    ax_deflection.set_ylabel("ΔY (mm)")
    ax_deflection.axhline(0, color='gray', linestyle='--', lw=1)
    ax_deflection.legend()
    ax_deflection.grid(True)
    fig_positions.tight_layout()

    fig_beam_analysis, (ax_moment, ax_shear) = plt.subplots(2, 1, figsize=(8, 7))
    moment_fit_plot, = ax_moment.plot([], [], 'g-', label='Moment (from Fit)')
    shear_fit_plot, = ax_shear.plot([], [], 'c-', label='Shear (from Fit)')
    
    for ax_bm in [ax_moment, ax_shear]:
        ax_bm.axhline(0, color='gray', linestyle='--')
        ax_bm.legend()
        ax_bm.grid(True)
        
    ax_moment.set_title("Bending Moment Diagram")
    ax_moment.set_ylabel("Moment (N·m)") # Using N.m
    ax_shear.set_title("Shear Force Diagram")
    ax_shear.set_ylabel("Shear Force (N)")    
    ax_shear.set_xlabel("Position along beam (mm)")
    fig_beam_analysis.tight_layout()
    plt.show(block=False)

    # --- Main Processing Loop ---
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
                calibration_info["active"] = True # Indicate calibration mode is active for one cycle
                print("Calibration mode activated. Ensure markers are in reference position.")
            if key_state['space_pressed']:
                key_state['space_pressed'] = False
                if current_circles:
                    locked_positions = [(c[0], c[1]) for c in current_circles]
                    print(f"Locked {len(locked_positions)} positions: {locked_positions}")
                    # Clear previous deflections when new positions are locked
                    deflections_mm.clear() 
                    ax_deflection.set_ylim(-10, 10) # Reset y-limit for deflection or make dynamic
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
            ax_positions.set_ylim(cropped_frame.shape[0], 0) # Y inverted
        else:
            live_scatter.set_offsets(np.empty((0,2)))
            live_line.set_data([], [])

        if locked_positions:
            lx, ly = zip(*locked_positions)
            locked_scatter.set_offsets(np.column_stack((lx, ly)))
            locked_line.set_data(lx, ly)
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
            locked_scatter.set_offsets(np.empty((0,2)))
            locked_line.set_data([],[])
            
        # --- Calculate and Plot Deflections ---
        if pixel_scale != 1.0 and locked_positions and len(current_circles) == len(locked_positions):
            deflections_mm = [(curr[1] - ref[1]) * pixel_scale for curr, ref in zip(current_circles, locked_positions)]
            
            marker_indices = range(len(deflections_mm))
            deflection_plot.set_data(marker_indices, deflections_mm)
            
            if deflections_mm:
                min_def, max_def = min(deflections_mm), max(deflections_mm)
                padding = (max_def - min_def) * 0.1 + 1 # Add 1mm padding minimum
                ax_deflection.set_ylim(min_def - padding, max_def + padding)
                ax_deflection.set_xlim(-0.5, len(deflections_mm) - 0.5)
            else:
                 ax_deflection.set_ylim(-10, 10) # Default if no deflections
        else:
            deflection_plot.set_data([],[])


        # --- Beam Analysis (Moment and Shear) ---
        # Ensure we have enough data points (at least 3 for a curve fit to make sense)
        # and that calibration has been done (pixel_scale is not its default 1.0)
        if pixel_scale != 1.0 and locked_positions and len(deflections_mm) >= 3:
            try:
                # Get x positions of the *locked* markers in mm from the first locked marker
                # Assuming the first locked marker is the "fixed end" or reference x=0 for beam theory
                # This uses the x-coordinates from the *locked_positions* for a stable beam length.
                x_coords_pixels_locked = np.array([pos[0] for pos in locked_positions])
                x_coords_mm_beam_axis = (x_coords_pixels_locked - x_coords_pixels_locked[0]) * pixel_scale

                y_deflections_mm = np.array(deflections_mm)

                # Convert to meters for physics formulas
                x_m = x_coords_mm_beam_axis / 1000.0
                y_m = y_deflections_mm / 1000.0
                
                beam_length_L_m = np.max(x_m) # Effective length of the tracked part of the beam

                if beam_length_L_m > 0 and EI_current_beam > 0:
                    # Fit the load P to the observed deflections using the theoretical model
                    # The curve_fit will find the 'P' that best matches the y_m data for given x_m
                    popt, pcov = curve_fit(
                        lambda x, P: cantilever_deflection_theory(x, P, EI_current_beam, beam_length_L_m),
                        x_m,
                        y_m,
                        p0=[1.0] # Initial guess for P in Newtons
                    )
                    fitted_load_P = popt[0]
                    print(f"Fitted Load P: {fitted_load_P:.2f} N")

                    # Calculate moment and shear based on this fitted load P
                    # M(x) = -P * (L - x)  (Moment for cantilever with load at end, x from fixed end)
                    # V(x) = -P            (Shear for cantilever with load at end)
                    moment_Nm_fit = -fitted_load_P * (beam_length_L_m - x_m)
                    shear_N_fit = np.full_like(x_m, -fitted_load_P) # Shear is constant

                    # Plotting (convert x back to mm for the plot axis)
                    moment_fit_plot.set_data(x_coords_mm_beam_axis, moment_Nm_fit)
                    shear_fit_plot.set_data(x_coords_mm_beam_axis, shear_N_fit)

                    for ax_bm, data in [(ax_moment, moment_Nm_fit), (ax_shear, shear_N_fit)]:
                        if data.size > 0:
                            min_val, max_val = np.min(data), np.max(data)
                            padding = (max_val - min_val) * 0.1 + 0.1 # Min 0.1 padding
                            ax_bm.set_ylim(min_val - padding, max_val + padding)
                            ax_bm.set_xlim(np.min(x_coords_mm_beam_axis), np.max(x_coords_mm_beam_axis))
                        else: # Default limits
                            ax_bm.set_xlim(0, KNOWN_DISTANCE_MM)
                            ax_bm.set_ylim(-10, 10 if ax_bm == ax_moment else -50, 50)
            except RuntimeError:
                print("Curve fit failed. Check data or model.")
            except Exception as e:
                print(f"Error in beam analysis: {e}")
        else: # Clear plots if not enough data
            moment_fit_plot.set_data([],[])
            shear_fit_plot.set_data([],[])


        # --- Update Figure Canvases ---
        fig_positions.canvas.draw_idle()
        fig_positions.canvas.flush_events()
        fig_beam_analysis.canvas.draw_idle()
        fig_beam_analysis.canvas.flush_events()

        # --- Display Calibration Text on Frame ---
        if calibration_info["counter"] > 0:
            cv2.putText(cropped_frame, calibration_info["text"], (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            calibration_info["counter"] -= 1

        cv2.imshow("Live Webcam Feed (Cropped) - Press 'q' to quit", cropped_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): # Fallback if pynput fails or for direct q
             break
  
    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close(fig_positions)
    plt.close(fig_beam_analysis)
    print("Camera and plots closed.")

# --- Tkinter GUI Setup ---
root = tk.Tk()
root.title("Beam Deflection Analyzer - Setup")

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

tk.Label(root, text="2. Select Beam Profile:", font=('Helvetica', 10, 'bold')).pack(pady=(10,0), anchor='w', padx=10)
beam_listbox = tk.Listbox(root, height=len(BEAM_LIST_NAMES), width=60, exportselection=False)
for beam_name in BEAM_LIST_NAMES:
    beam_listbox.insert(tk.END, beam_name)
beam_listbox.pack(padx=10, pady=5, fill=tk.X)
if BEAM_LIST_NAMES: # Select first beam by default
    beam_listbox.select_set(0)


tk.Label(root, text="Instructions:", font=('Helvetica', 10, 'bold')).pack(pady=(10,0), anchor='w', padx=10)
instructions_text = (
    "- Press 'c' to initiate calibration (ensure markers are at known distance).\n"
    "- Press 'SPACE' to lock current marker positions as reference.\n"
    "- Press 'q' in the feed window or GUI to quit."
)
tk.Label(root, text=instructions_text, justify=tk.LEFT).pack(padx=10, pady=5, anchor='w')


start_button = tk.Button(root, text="Start Analysis", command=start_camera_processing, font=('Helvetica', 12, 'bold'), bg='lightblue')
start_button.pack(pady=20, padx=10, fill=tk.X)

# Start keyboard listener thread (daemon so it exits with main)
kbd_listener_thread = Thread(target=key_listener_thread, daemon=True)
kbd_listener_thread.start()

root.mainloop()