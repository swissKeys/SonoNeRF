import os

# The path to your folder containing the images
folder_path = "data/rawdata/Sononerf_Data_1/images"

# List all the images in the folder
image_filenames = os.listdir(folder_path)

# Sort the filenames
image_filenames.sort()

# Iterate over the sorted filenames
for i, filename in enumerate(image_filenames):
    # Create the new filename. 
    # Use str.zfill to make sure the number is 4 digits with leading zeros.
    new_filename = f"Image_{str(i+1).zfill(4)}.jpg"

    # Create the full paths to the old and new filenames
    old_file_path = os.path.join(folder_path, filename)
    new_file_path = os.path.join(folder_path, new_filename)

    # Rename the file
    os.rename(old_file_path, new_file_path)
