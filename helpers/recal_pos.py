import numpy as np

# Read the CSV file
data = np.loadtxt('data/preprocessed_data/poses_bounds_estimated.csv', delimiter=',')

# Replace the last two values in each line with 10 and 297
data[:, -2:] = 10, 297

# Save as a new CSV file
np.savetxt('poses_bounds.csv', data, delimiter=',')

# Save as a new NumPy .npy file
np.save('poses_bounds.npy', data)