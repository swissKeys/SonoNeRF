import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

folder_path = "data/rawdata/Sononerf_Data_1/rebecca-long-1_jpg"

# Get a list of image file names in the folder
image_files = sorted(os.listdir(folder_path))

# Read and process the images
image_stack = []
for image_file in image_files:
    # Load image and convert to RGBA
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path).convert("RGBA")

    # Rotate the image by -90 degrees
    image = image.rotate(90, expand=True)

    # Extract the image data
    image_data = np.array(image)

    # Make the entire image 50% transparent
    image_data[:, :, 3] = image_data[:, :, 3] // 2

    # Append the modified image to the image stack
    image_stack.append(image_data)

# Create a figure and 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set the aspect ratio of the plot
ax.set_box_aspect([0.711, 1.140, 1])

# Loop over the image stack and plot each image
y = 0
counter = 0
for image_data in image_stack:
    print(counter)
    depth, width, _ = image_data.shape
    x, z = np.meshgrid(np.arange(width), np.arange(depth))
    ax.plot_surface(x/3, np.ones_like(x) * y, z, facecolors=image_data/255.0, rstride=1, cstride=1)
    y += 4
    counter += 1

# Show grid lines
ax.grid(True)

# Show axis labels
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Y')

# Show the plot
plt.show()
