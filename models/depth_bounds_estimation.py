import cv2
import numpy as np
import os

def depth_bounds_estimation(frames_path):
    # Load frames
    frames = [cv2.imread(os.path.join(frames_path, f), cv2.IMREAD_GRAYSCALE) for f in sorted(os.listdir(frames_path)) if f.endswith('.jpg')]

    # Step 3: Combined Otsu's method and Adaptive Gaussian Thresholding
    thresholded_frames = []

    for frame in frames:
        # Apply Otsu's thresholding
        _, otsu_thresh = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply adaptive Gaussian thresholding
        adaptive_thresh = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Combine the results
        combined_thresh = cv2.bitwise_and(otsu_thresh, adaptive_thresh)
        thresholded_frames.append(combined_thresh)

    # Step 4: Calculate depth bounds
    depth_bounds = []

    for frame in thresholded_frames:
        # Find the non-zero pixel coordinates
        nonzero_y, nonzero_x = np.nonzero(frame)

        # If there are no non-zero pixels, set the depth bounds to zero
        if len(nonzero_y) == 0:
            depth_bounds.append((0, 0))
        else:
            # Calculate the near and far depth bounds based on the minimum and maximum y-coordinates
            near_depth = np.min(nonzero_y)
            far_depth = np.max(nonzero_y)
            depth_bounds.append((near_depth, far_depth))

    return depth_bounds