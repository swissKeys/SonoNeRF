import numpy as np

data_processed = []

with open("data/rawdata/Sononerf_Data_1/poses.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        values = line.split()  # Assuming the values are space-separated
        if len(values) >= 9:  # Check if there are enough values in this line
            # Ignore the first two values and keep the rest
            data_processed.append(values[2:])

# If you want to convert the values to float:
for i in range(len(data_processed)):
    data_processed[i] = [float(val) for val in data_processed[i]]

# Assume near_bound and far_bound are your determined values
near_bound = 0
far_bound = 8

# Append the depth bounds to each data line
for i in range(len(data_processed)):
    data_processed[i] += [near_bound, far_bound]

# Convert the list to a numpy array
data_processed_np = np.array(data_processed)

# Save the numpy array to a file
np.save('poses_bounds.npy', data_processed_np)
