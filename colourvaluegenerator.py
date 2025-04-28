import numpy as np

orange_lower = np.array([0, 51, 200])
orange_upper = np.array([10, 255, 255])
green_lower = np.array([86, 80, 50])
green_upper = np.array([95, 255, 150])

np.savez("colourvalues.npz", 
         orange_lower=orange_lower, orange_upper=orange_upper, 
         green_upper=green_upper, green_lower=green_lower
         )
print("Saved!")