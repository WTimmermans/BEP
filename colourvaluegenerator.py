import numpy as np

orange_upper = np.array([10, 255, 255])
orange_lower = np.array([0, 50, 200])

np.savez("colourvalues.npz", orange_lower=orange_lower, orange_upper=orange_upper)
print("Saved!")