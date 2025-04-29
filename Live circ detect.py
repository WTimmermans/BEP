# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 14:37:42 2025

ChatGPT provided code for live webcam feed circle detection.

@author: Steffen
"""

import cv2
import numpy as np

# Open webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if webcam is opened
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Blur to reduce noise before applying Hough Transform
    gray = cv2.medianBlur(gray, 5)
    
    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,           # Inverse ratio of accumulator resolution to image resolution
        minDist=50,       # Minimum distance between circle centers
        param1=50,        # Higher threshold for Canny edge detector
        param2=30,        # Accumulator threshold for circle detection (smaller -> more false circles)
        minRadius=1,     # Minimum circle radius
        maxRadius=15     # Maximum circle radius
    )

    # If some circles are detected, draw them
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw outer circle
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw center of the circle
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

    # Display the result
    cv2.imshow('Hough Circle Detection', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
cv2.destroyAllWindows()
