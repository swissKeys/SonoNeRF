# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import os
import pathlib
import sys
import cv2
from tensorflow.keras.models import load_model
from scipy.spatial.transform import Rotation
import pandas as pd

sys.stdout.flush()
from src.llff_preprocessing import gen_poses


def video_preprocessing(args):

    video_path = args.input
    output_folder = args.output
    ffmpeg_path = args.ffmpeg_path

    # extract frames
    images_folder = os.path.join(output_folder, "images/")
    create_folder(images_folder)
    from subprocess import run, check_output, STDOUT, DEVNULL

    command = ""
    # command += "-i " + video_path + " -f image2 -qscale:v 1 -qmin 1 " + images_folder + "image%05d.jpg" # highest quality and all images.
    command += (
        "-i "
        + video_path
        + " -f image2 -qscale:v 2 -vf fps="
        + str(args.fps)
        + " "
        + images_folder
        + "image%05d.png"
    )
    # command += "-i " + video_path + ' -f image2 -qscale:v 2 -vf "fps=' + str(args.fps) + ', crop=in_w:3/4*in_h:0:in_h/4" ' + images_folder + "image%05d.png" # crop
    print(command, flush=True)
    try:
        ffmpeg_output = check_output([ffmpeg_path] + command.split(" "), stderr=STDOUT)
    except:
        run(ffmpeg_path + " " + command)

    # take care of failed frames
    failed_frames_folder = os.path.join(output_folder, "images_failed/")
    if os.path.exists(failed_frames_folder):
        failed_frame_names = os.listdir(failed_frames_folder)
        print(
            "detected failed frames, will delete: " + str(failed_frame_names),
            flush=True,
        )
        [
            os.remove(os.path.join(images_folder, failed_frame))
            for failed_frame in failed_frame_names
        ]

    # create videos using ffmpeg
    print("creating full-resolution RGB video...", flush=True)
    command = ""
    command += (
        "-framerate " + str(args.fps) + " -i " + images_folder + "image%05d.png -y "
    )  # -y overwrites existing files automatically
    command += os.path.join(output_folder, "rgb_scene_fullres.mp4")
    try:
        ffmpeg_output = check_output([ffmpeg_path] + command.split(" "), stderr=STDOUT)
    except:
        run(ffmpeg_path + " " + command)

    # print("creating downsampled RGB video...", flush=True)
    # command = ""
    # command += "-i " + os.path.join(output_folder, "rgb_scene_fullres.mp4") + ' -vf scale="iw/1:ih/2"' + " -y "
    # command += os.path.join(output_folder, "rgb_scene_downsampled.mp4")
    # ffmpeg_output = check_output([ffmpeg_path] + command.split(" "), stderr=STDOUT)



def create_folder(folder):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

def preprocess(args):

    # output folder
    if args.output is None:
        if os.path.isfile(args.input):
            input_folder, input_file = os.path.split(args.input)
            input_name, input_extension = os.path.splitext(input_file)
            args.output = os.path.join(input_folder, input_name)
        else:
            args.output = args.input
    create_folder(args.output)

    # video extraction
    if os.path.isfile(args.input):
        video_preprocessing(args)
        args.input = args.output
        # get camera poses by running colmap but will probably change to soemthign that can ahndle ultrasound images
        crop(args)
        print("cropped images")
        applyFilters(args)
        print("applied filters")
        args.input = os.path.join(args.output)
        #genPoses(args)
        print("estimated poses")

def crop(args):
    # Define the path to the directory containing the ultrasound images
    img_dir = os.path.join(args.output, "images")
    images_folder = os.path.join(args.output, "crop_images/")
    create_folder(images_folder)

    # Define the coordinates of the ROI
    x = 100 # x-coordinate of the top-left corner of the ROI
    y = 80 # y-coordinate of the top-left corner of the ROI
    w = 310 # width of the ROI
    h = 300 # height of the ROI
    # Create a directory to store the cropped ROIs
    roi_dir = os.path.join(args.output, "crop_images")
    # Loop over all the ultrasound images
    for filename in os.listdir(img_dir):
            # Load the ultrasound image
        img = cv2.imread(os.path.join(img_dir, filename), cv2.IMREAD_GRAYSCALE)

        # Crop the ROI from the denoised image
        roi = img[y:y+h, x:x+w]
        # Save the cropped ROI to a file
        roi_filename = os.path.splitext(filename)[0] + "_crop.jpg"
        roi_path = os.path.join(roi_dir, roi_filename)
        cv2.imwrite(roi_path, roi)

def genPoses(args):
    #load model
    model = load_model("models/ultrasound_pose_estimator.h5")
    filterd_image_folder = os.path.join(args.output, "filtered_images")
    image_files = sorted(os.listdir(filterd_image_folder))
    image_size = (128, 128)  # Resize the images to a smaller size
    images = []
    for img_path in image_files:
        img = cv2.imread(os.path.join(filterd_image_folder, img_path))
        img_resized = cv2.resize(img, image_size)
        images.append(img_resized) 
    images = np.array(images)
    #predict position and orientation of probe for each frame
    predictions = model.predict(images)

    positions = predictions[:, :3]        # Shape: (N, 3)
    quaternions = predictions[:, 3:]      # Shape: (N, 4)

    rotations = Rotation.from_quat(quaternions)
    rotation_matrices = rotations.as_matrix()

    N = predictions.shape[0]
    pose_matrices = np.zeros((N, 4, 4))

    pose_matrices[:, :3, :3] = rotation_matrices
    pose_matrices[:, :3, 3] = positions 

    pose_matrices[:, 3, 3] = 1

    pose_matrices_flat = pose_matrices.reshape(N, -1)
    pose_matrices_df = pd.DataFrame(pose_matrices_flat)
    pose_matrices_df.to_csv(os.path.join(args.output, "pose_matrices.csv"), index=False)
    np.save(os.path.join(args.output, "pose_matrices.npy"), pose_matrices)

def applyFilters(args):
    # Create folders
    img_dir = os.path.join(args.output, "crop_images")
    images_folder = os.path.join(args.output, "filtered_images/")
    create_folder(images_folder)
    filtered_dir = os.path.join(args.output, "filtered_images")
    def enhance_contrast(img):
        # Check if the input image is grayscale
        if len(img.shape) == 2:
            # If the image is grayscale, apply histogram equalization directly
            eq = cv2.equalizeHist(img)
        else:
            # If the image is color, convert it to grayscale and then apply histogram equalization
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            eq = cv2.equalizeHist(gray)

        # Convert the equalized image back to color
        eq_color = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

        return eq_color
    
    for filename in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir, filename), cv2.IMREAD_GRAYSCALE)
        # Calculate the local mean and variance of the image for denoising
        mean, var = cv2.meanStdDev(img)
        mean = mean[0][0]
        var = var[0][0]
        # Calculate the optimal denoising parameters based an mean and variance
        h = 3 * var
        searchWindowSize = int(max(5, 2 * np.sqrt(var)))
        templateWindowSize = int(max(3, np.sqrt(var)))
        img = cv2.fastNlMeansDenoising(img, h, searchWindowSize, templateWindowSize)
        #img = enhance_contrast(denoised)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # Save the cropped ROI to a file
        filtered_filename = os.path.splitext(filename)[0] + "_filterd.jpg"
        filtered_path = os.path.join(filtered_dir, filtered_filename)
        cv2.imwrite(filtered_path, img)


if __name__ == "__main__":

    import configargparse

    parser = configargparse.ArgumentParser()
    # mandatory arguments
    parser.add_argument(
        "--input",
        type=str,
        help='input. can be a video file or folder that contains a subfolder named "images", which contains images. e.g. set to foo/bar if images are in foo/bar/images/image0.png',
    )
    # optional custom paths
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help='custom output folder. similar to --input, needs to be foo/bar such that subfolders like "images" can be created as foo/bar/images/',
    )
    parser.add_argument(
        "--colmap_matching",
        type=str,
        default="sequential_matcher",
        help='"sequential_matcher" (default. for temporally ordered input, e.g. video) or "exhaustive_matcher" (each image is matched with every other image).',
    )
    parser.add_argument(
        "--ffmpeg_path",
        type=str,
        default="ffmpeg",
        help="path to ffmpeg executable. only used for video input.",
    )
    # video input
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="when using video input, the frame rate at which images should be extracted from the video",
    )
    # apply computed lens distortion to undistort the input

    args = parser.parse_args()

    preprocess(args)