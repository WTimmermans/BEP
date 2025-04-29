"""
Created on Mon Apr 28 13:07:24 2025

Code to recognise circles in a (static) image.

@author: Steffen
"""

import sys
import cv2
import numpy as np



def main(argv):
    ## [load]
    default_file = 'ballonnen.png'
    filename = argv[0] if len(argv) > 0 else default_file

    # Loads an image
    src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_COLOR)

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1
    ## [load]

    ## [convert_to_gray]
    # Convert it to gray
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ## [convert_to_gray]

    ## [reduce_noise]
    # Reduce the noise to avoid false circle detection
    gray = cv2.medianBlur(gray, 5)
    
    # Show Gray scale pic
    cv2.imshow("Gray scale", gray)
    
    ## [reduce_noise]

    ## [houghcircles]
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=0, maxRadius=100)
    ## [houghcircles]

    ## [draw]
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(src, center, radius, (255, 0, 255), 3)
    ## [draw]

    ## [display]
    cv2.imshow("detected circles", src)
    cv2.waitKey(0)
    ## [display]

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])