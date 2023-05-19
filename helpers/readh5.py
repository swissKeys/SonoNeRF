import os
import h5py
import matplotlib.pyplot as plt
import numpy as np

# The path to the directory containing the .h5 files
directory = 'data/rawdata/Sononerf_Data_1/rebecca-short-1.3'

# The path to the directory where the output images will be saved
output_dir = 'data/rawdata/Sononerf_Data_1/rebecca-short-1.3_jpg'


# Get a list of all .h5 files in the directory
files = [f for f in os.listdir(directory) if f.endswith('.h5')]

# Iterate over each file
for file in files:
    # Open the .h5 file
    with h5py.File(os.path.join(directory, file), 'r') as f:
        # Get the 'Tissue' dataset
        data = f['Tissue'][:]

    # Squeeze out the singleton dimensions
    data = np.squeeze(data)

        # Check if data is 2D
    if data.ndim != 2:
        print(f"Skipping file {file} because its data is not 2D.")
        continue

    # Display the image data
    plt.imshow(data, cmap='gray')  # or any other color map that suits your data
    plt.axis('off')

    # Save the figure
    output_file = os.path.join(output_dir, f'{os.path.splitext(file)[0]}.jpg')
    plt.savefig(output_file, bbox_inches='tight', pad_inches = 0)

    # Clear the current figure to free memory
    plt.clf()