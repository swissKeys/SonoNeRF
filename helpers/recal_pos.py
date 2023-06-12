import os
from PIL import Image, ImageOps

# Directory paths
input_dir = 'data/preprocessed_data/images_sononerf'
output_dir = 'data/preprocessed_data/images_sononerf_black'

# Frame thickness in pixels
frame_thickness = 100

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# List all files in the directory
files = os.listdir(input_dir)

for file_name in files:
    # Only process if file is an image (you may need to adjust this according to your image types)
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Open image
        image_path = os.path.join(input_dir, file_name)
        img = Image.open(image_path)
        
        # Add black border
        img_with_border = ImageOps.expand(img, border=frame_thickness, fill='black')
        
        # Save to output directory
        output_path = os.path.join(output_dir, file_name)
        img_with_border.save(output_path)
