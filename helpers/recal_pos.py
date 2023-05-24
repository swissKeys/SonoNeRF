import numpy as np

# Open the text file
with open('processed_data_file.txt', 'r') as file:
    lines = file.readlines()

# Add "0 8" to each line
lines_with_depthbounds = [line.strip() + " 0 8\n" for line in lines]

# Convert to NumPy array
data = np.loadtxt(lines_with_depthbounds)

# Save as CSV file
np.savetxt('poses_bounds.csv', data, delimiter=',')

# Save as NumPy .npy file
np.save('poses_bounds.npy', data)
