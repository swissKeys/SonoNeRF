# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import os
import pathlib
import sys

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
        gen_poses(args.input, args.colmap_matching)


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