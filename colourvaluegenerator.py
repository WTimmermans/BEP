### This subscript contains all the lower and upper HSV colour values
## for our sticker set. Can be changed to fit the situation.

import numpy as np

### Define colour lower and upper value ###

yellow_lower = np.array([15, 151, 151])
yellow_upper = np.array([26, 255, 255])

lightgreen_lower = np.array([41, 14, 99])
lightgreen_upper = np.array([62, 255, 255])

darkgreen_lower = np.array([94, 141, 0])
darkgreen_upper = np.array([100, 255, 135])

aquamarine_lower = np.array([90, 111, 0])
aquamarine_upper = np.array([101, 141, 198])

turquoise_lower = np.array([90, 61, 173])
turquoise_upper = np.array([110, 255, 248])

lightblue_lower = np.array([94, 101, 180])
lightblue_upper = np.array([104, 255, 255])

darkblue_lower = np.array([102, 139, 0])
darkblue_upper = np.array([111, 255, 165])

purple_lower = np.array([124, 45, 99])
purple_upper = np.array([167, 255, 255])

violet_lower = np.array([168, 85, 121])
violet_upper = np.array([179, 255, 255])

red_lower = np.array([0, 199, 167])
red_upper = np.array([7, 255, 255])

orange_lower = np.array([10, 179, 180])
orange_upper = np.array([18, 255, 255])

pink_lower = np.array([0, 76, 182])
pink_upper = np.array([13, 128, 219])

brown_lower = np.array([4, 0, 40])
brown_upper = np.array([13, 137, 132])

### Save values as .npz ###

np.savez("colourvalues.npz", 
         yellow_lower=yellow_lower, yellow_upper=yellow_upper,
         lightgreen_lower=lightgreen_lower, lightgreen_upper=lightgreen_upper,
         darkgreen_lower=darkgreen_lower, darkgreen_upper=darkgreen_upper,
         aquamarine_lower=aquamarine_lower, aquamarine_upper=aquamarine_upper,
         turquoise_lower=turquoise_lower, turquoise_upper=turquoise_upper,
         lightblue_lower=lightblue_lower, lightblue_upper=lightblue_upper,
         darkblue_lower=darkblue_lower, darkblue_upper=darkblue_upper,
         purple_lower=purple_lower, purple_upper=purple_upper,
         violet_lower=violet_lower, violet_upper=violet_upper,
         red_lower=red_lower, red_upper=red_upper,
         orange_lower=orange_lower, orange_upper=orange_upper,
         pink_lower=pink_lower, pink_upper=pink_upper,
         brown_lower=brown_lower, brown_upper=brown_upper
)

print("Saved!")
