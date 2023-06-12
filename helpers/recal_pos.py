import numpy as np

# Read the CSV file
data = np.loadtxt('data/preprocessed_data/pose_bounds_old.csv', delimiter=',')

# Save as a new NumPy .npy file
np.save('poses_bounds.npy', data)