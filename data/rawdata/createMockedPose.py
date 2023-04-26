import numpy as np
import quaternion

num_samples = 2903

frame_numbers = np.arange(1, num_samples + 1).reshape(-1, 1)

# Generate linearly spaced positions
start_position = np.array([-10, -10, 0])  # Starting x, y, z coordinates
end_position = np.array([10, 10, 10])  # Ending x, y, z coordinates
linear_positions = np.linspace(start_position, end_position, num_samples)

# Generate linearly spaced orientations
start_quaternion = quaternion.from_euler_angles(0, 0, 0)
end_quaternion = quaternion.from_euler_angles(0, np.pi, 0)
alpha = np.linspace(0, 1, num_samples)
linear_orientations = np.array([quaternion.slerp(start_quaternion, end_quaternion, 0, 1, a).components for a in alpha])

# Concatenate positions, frame numbers and orientations
positions_orientations = np.concatenate((frame_numbers, linear_positions, linear_orientations), axis=1)
# Save as CSV
header = "frame_number,position_x,position_y,position_z,orientation_w,orientation_x,orientation_y,orientation_z"
np.savetxt("synthetic_positions_orientations.csv", positions_orientations, delimiter=",", header=header, comments='')

# Save as NumPy file
np.save("synthetic_positions_orientations.npy", positions_orientations)
