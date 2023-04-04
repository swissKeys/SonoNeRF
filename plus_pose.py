import plus
import numpy as np

def estimate_camera_poses(config_file, calibration_file, sweep_file):
    # Initialize the PLUS library and load the configuration
    plus.initialize()
    plus_config = plus.OpenIGTLinkDeviceProxyConfiguration()
    plus_config.SetConfigurationFilePath(config_file)
    device = plus.OpenIGTLinkDeviceProxy()
    device.SetTypeWithName("PlusServer", plus_config)
    device.Connect()

    # Load the calibration phantom configuration
    calibration = plus.Calibration()
    calibration.LoadFromFile(calibration_file)

    # Load the ultrasound sweep data
    sweep = plus.Sweep()
    sweep.Load(sweep_file)

    # Configure the Ultrasound Image Calibration and Tracking module
    tracker = plus.UltrasoundImageCalibrationAndTracking()
    tracker.SetDevice(device)
    tracker.SetCalibration(calibration)
    tracker.SetSweep(sweep)

    # Estimate the camera pose for each ultrasound image in the sweep
    num_images = sweep.GetNumberOfFrames()
    poses = np.zeros((num_images, 6))
    for i in range(num_images):
        tracker.Update()
        pose = tracker.GetProbePositionOrientation()
        poses[i,:] = pose

    # Save the camera poses to a file
    np.savetxt("camera_poses.txt", poses)
    
    return poses