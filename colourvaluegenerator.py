import numpy as np

yellow_lower = np.array([])
yellow_upper = np.array([])

lightgreen_lower = np.array([])
lightgreen_upper = np.array([])

darkgreen_lower = np.array([])
darkgreen_upper = np.array([])

aquamarine_lower = np.array([])
aquamarine_upper = np.array([])

turquoise_lower = np.array([])
turquoise_upper = np.array([])

lightblue_lower = np.array([])
lightblue_upper = np.array([])

darkblue_lower = np.array([])
darkblue_upper = np.array([])

purple_lower = np.array([])
purple_upper = np.array([])

violet_lower = np.array([])
violet_upper = np.array([])

red_lower = np.array([])
red_upper = np.array([])

orange_lower = np.array([])
orange_upper = np.array([])

pink_lower = np.array([])
pink_upper = np.array([])

brown_lower = np.array([])
brown_upper = np.array([])

gray_lower = np.array([])
gray_upper = np.array([])

white_lower = np.array([])
white_upper = np.array([])

black_lower = np.array([])
black_upper = np.array([])

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
         brown_lower=brown_lower, brown_upper=brown_upper,
         gray_lower=gray_lower, gray_upper=gray_upper,
         white_lower=white_lower, white_upper=white_upper,
         black_lower=black_lower, black_upper=black_upper
)

print("Saved!")
