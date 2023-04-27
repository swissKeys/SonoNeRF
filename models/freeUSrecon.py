import numpy as np
import torch
import os
from PIL import Image
import torch.nn as nn
from pathlib import Path
import csv
import torch
import numpy as np
# import torchvision.models.resnet as resnet
from models.networks import resnet
from models.networks import resnext
import os
from os import path
# from functions import mahalanobis
from models.networks import generators
from models.networks import mynet
from models.networks import p3d
from models.networks import densenet


def load_images(folder_path, target_size=(224, 224)):
    image_files = sorted(os.listdir(folder_path))
    images = []
    for image_file in image_files:
        if image_file.endswith(".jpg"):
            image_path = os.path.join(folder_path, image_file)
            img = Image.open(image_path)
            img = img.resize(target_size, Image.ANTIALIAS)  # Resize the image
            img_array = np.asarray(img, dtype=np.float32)  # Convert to float32
            images.append(img_array)
    images = np.array(images)
    images = np.expand_dims(images, axis=1)  # Add channel dimension
    return images


def define_model(model_type, pretrained_path,
                 input_type, output_type, neighbour_slice):
    if input_type == 'diff_img':
        input_channel = neighbour_slice - 1
    else:
        input_channel = neighbour_slice

    if model_type == 'prevost':
        model_ft = generators.PrevostNet()
    elif model_type == 'resnext50':
        model_ft = resnext.resnet50(sample_size=2, sample_duration=16, cardinality=32)
        model_ft.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 7, 7),
                                   stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
    elif model_type == 'resnext101':
        model_ft = resnext.resnet101(sample_size=2, sample_duration=16, cardinality=32)
        model_ft.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 7, 7),
                                   stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        # model_ft.conv1 = nn.Conv3d(neighbour_slice, 64, kernel_size=7, stride=(1, 2, 2),
        #                            padding=(3, 3, 3), bias=False)
    elif model_type == 'resnet152':
        model_ft = resnet.resnet152(pretrained=True)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
    elif model_type == 'resnet101':
        model_ft = resnet.resnet101(pretrained=True)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
    elif model_type == 'resnet50':
        model_ft = resnet.resnet50(pretrained=True)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
    elif model_type == 'resnet34':
        model_ft = resnet.resnet34(pretrained=False)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
    elif model_type == 'resnet18':
        model_ft = resnet.resnet18(pretrained=True)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
    elif model_type == 'mynet':
        model_ft = mynet.resnet50(sample_size=2, sample_duration=16, cardinality=32)
        model_ft.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 7, 7),
                                   stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
    elif model_type == 'mynet2':
        model_ft = generators.My3DNet()
    elif model_type == 'p3d':
        model_ft = p3d.P3D63()
        model_ft.conv1_custom = nn.Conv3d(1, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                                          padding=(0, 3, 3), bias=False)
    elif model_type == 'densenet121':
        model_ft = densenet.densenet121()
    else:
        print('network type of <{}> is not supported, use original instead'.format(network_type))
        model_ft = generators.PrevostNet()

    num_ftrs = model_ft.fc.in_features

    if model_type == 'mynet':
        num_ftrs = 384
    elif model_type == 'prevost':
        num_ftrs = 576

    if output_type == 'average_dof' or output_type == 'sum_dof':
        # model_ft.fc = nn.Linear(128, 6)
        model_ft.fc = nn.Linear(num_ftrs, 6)
    else:
        # model_ft.fc = nn.Linear(128, (neighbour_slice - 1) * 6)
        model_ft.fc = nn.Linear(num_ftrs, (neighbour_slice - 1) * 6)



    # if args.training_mode == 'finetune':
    #     model_path = path.join(results_dir, args.model_filename)
    #     if path.isfile(model_path):
    #         print('Loading model from <{}>...'.format(model_path))
    #         model_ft.load_state_dict(torch.load(model_path))
    #         print('Done')
    #     else:
    #         print('<{}> not exists! Training from scratch...'.format(model_path))

    if pretrained_path:
        if path.isfile(pretrained_path):
            print('Loading model from <{}>...'.format(pretrained_path))
            model_ft.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
            # model_ft.load_state_dict(torch.load(pretrained_path))
            print('Done')
        else:
            print('<{}> not exists! Training from scratch...'.format(pretrained_path))
    else:
        print('Train this model from scratch!')

    #model_ft.cuda()
    device = torch.device("cpu")
    model_ft = model_ft.to(device)
    print('define model device {}'.format(device))
    return model_ft

class ApplyModel:
    def __init__(self, model_ft, input_data, cam_cali_mat, device, frames_num, neighbour_slice):
        self.model_ft = model_ft
        self.input_data = input_data
        self.cam_cali_mat = cam_cali_mat
        self.device = device
        self.frames_num = frames_num
        self.neighbour_slice = neighbour_slice
        self.estimate_poses()

    def estimate_poses(self):
        self.model_ft.eval()
        self.model_ft.to(self.device)

        print("Input data shape:", self.input_data.shape)

        batch_size = 1  # Process one image at a time
        n_images = len(self.input_data)
        results = []

        for i in range(0, n_images, batch_size):
            print(f"Processing image {i + 1} of {n_images}")
            batch = self.input_data[i:i+batch_size]
            batch = np.expand_dims(batch, axis=1)  # Add channel dimension
            input_tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                output = self.model_ft(input_tensor)

            results.append(output[0].detach().cpu().numpy())
        print(f"Number of estimated poses: {len(results)}")

        # Check the dimensions and data type of each item in the results list
        for i, result in enumerate(results):
            print(f"Pose {i+1}:")
            print(f"  Shape: {result.shape}")
            print(f"  Data type: {result.dtype}")
            print(f"  Pose values: {result}")

        estimated_matrices = []

        def compute_absolute_matrices(estimated_matrices):
            absolute_matrices = [np.eye(4)]  # Initialize with the identity matrix for the first frame

            for M_est in estimated_matrices:
                M_abs = np.matmul(absolute_matrices[-1], M_est)
                absolute_matrices.append(M_abs)

            return absolute_matrices


        def create_rotation_matrix(ax, ay, az):
            # Convert angles to radians
            ax_rad, ay_rad, az_rad = np.radians(ax), np.radians(ay), np.radians(az)

            # Calculate rotation matrices for each axis
            Rx = np.array([[1, 0, 0],
                        [0, np.cos(ax_rad), -np.sin(ax_rad)],
                        [0, np.sin(ax_rad), np.cos(ax_rad)]])

            Ry = np.array([[np.cos(ay_rad), 0, np.sin(ay_rad)],
                        [0, 1, 0],
                        [-np.sin(ay_rad), 0, np.cos(ay_rad)]])

            Rz = np.array([[np.cos(az_rad), -np.sin(az_rad), 0],
                        [np.sin(az_rad), np.cos(az_rad), 0],
                        [0, 0, 1]])

            # Combine the rotation matrices
            R_est = np.matmul(Rz, np.matmul(Ry, Rx))
            return R_est

        for theta in results:
            tx, ty, tz, ax, ay, az = theta[0]  # Use theta[0] instead of theta to extract values
            R_est = create_rotation_matrix(ax, ay, az)  # You'll need to implement this function
            T_est = np.array([tx, ty, tz]).reshape(3, 1)
            M_est = np.hstack((R_est, T_est))
            M_est = np.vstack((M_est, np.array([0, 0, 0, 1])))
            estimated_matrices.append(M_est)

        absolute_matrices = compute_absolute_matrices(estimated_matrices)
        
        self.result_params = absolute_matrices

# Load the pretrained model

def run_pose_estimator(folder_path, model_string='mc72', model_folder='pretrained_networks', output_filename='output.csv', device_no=0):
    model_folder = 'pretrained_networks'
    model_path = Path(model_folder, f'3d_best_Generator_{model_string}.pth')

    network_type = 'resnext50'
    input_type = 'org_img'
    output_type = 'average_dof'
    neighbour_slice = 4  # The number of neighboring slices
    device = torch.device("cpu")
    images = load_images(folder_path)

    model_ft = define_model(network_type,
                                        model_path,
                                        input_type,
                                        output_type,
                                        neighbour_slice)
    model_ft = model_ft.to(device)

    # Parameters from the ultrasound system
    resolution = 0.71  # Spatial resolution in mm/pixel
    center_frequency = 2.22  # Center frequency in MHz
    # Image dimensions in pixels
    image_width_pixels = 640
    image_height_pixels = 480

    # Physical dimensions of the image in millimeters
    image_width_mm = image_width_pixels * resolution
    image_height_mm = image_height_pixels * resolution

    # Calculate the field of view (FOV) in radians
    FOV_horizontal = 2 * np.arctan((image_width_mm * 0.5) / (image_width_pixels * 0.5))
    FOV_vertical = 2 * np.arctan((image_height_mm * 0.5) / (image_height_pixels * 0.5))

    # Calculate the focal length
    f_x = (image_width_pixels * 0.5) / np.tan(FOV_horizontal * 0.5)
    f_y = (image_height_pixels * 0.5) / np.tan(FOV_vertical * 0.5)

    # Estimate the optical center
    c_x = image_width_pixels * 0.5
    c_y = image_height_pixels * 0.5

    cam_cali_mat = np.array([[f_x, 0, c_x],
                            [0, f_y, c_y],
                            [0, 0, 1]])

    device = torch.device("cpu")
    frames_num = len(images)
    neighbour_slice = 4 # The number of neighboring slices

    # Apply the model to the new dataset
    apply_model = ApplyModel(model_ft, images, cam_cali_mat, device, frames_num, neighbour_slice)
    estimated_poses = apply_model.result_params

    estimated_poses_flattened = []

    for mat in estimated_poses:
        flattened_mat = mat.flatten()
        estimated_poses_flattened.append(flattened_mat)

    estimated_poses_flattened = np.array(estimated_poses_flattened)

    np.save('estimated_poses_flattened.npy', estimated_poses_flattened)

    with open('output.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        
        for row in estimated_poses_flattened:
            csv_writer.writerow(row)