% field_ii_demo.m

% Add Field II folder to MATLAB path
addpath('/Users/rebeccaadmin/Downloads/Field_II_ver_3_30_mac')

% Initialize Field II
field_init(0);

% Define the transducer parameters
f0 = 3.5e6; % Center frequency (3.5 MHz)
fs = 100e6; % Sampling frequency (100 MHz)
c = 1540; % Speed of sound (m/s) speed at which ultrasound waves propagate through the medium, typically soft tissue.
n_elements = 128; % Number of elements crytslas
element_width = 0.15e-3; % Element width (0.15 mm)crytslas
element_height = 5e-3; % Element height (5 mm)crytslas
kerf = 0.05e-3; % Kerf width (0.05 mm) Space bewteen crytslas
focus = [0, 0, 50e-3]; % Focus at 50 mm depth

% Create the linear array transducer
Th = xdc_linear_array(n_elements, element_width, element_height, kerf, 1, 1, focus);

% Set the excitation signal
excitation_signal = sin(2 * pi * f0 * (0:1 / fs:2 / f0));
xdc_excitation(Th, excitation_signal);

% Set the impulse response
impulse_response = sin(2 * pi * f0 * (0:1 / fs:2 / f0));
impulse_response = impulse_response .* hann(length(impulse_response))';
xdc_impulse(Th, impulse_response);


% Define image properties for the liver
image_depth = 150e-3; % Image depth (150 mm)
image_width = 80e-3; % Image width (80 mm)
n_lines = 200; % Number of lines
d_x = image_width / n_lines; % Lateral distance between lines

% Load the liver mask and volume images
liver_mask_path = 'CT_data/dataset_6/dataset_6/segmentation-0_livermask_60.png';
volume_path = 'CT_data/dataset_6/dataset_6/volume-0_60.png';
liver_mask = imread(liver_mask_path);

% Upscale the liver mask to match the scatterer_positions size

scaled_liver_mask = imresize(liver_mask, [round(fs * image_depth / c), n_lines]);
scaled_liver_mask = rgb2gray(scaled_liver_mask);
scaled_liver_mask = transpose(scaled_liver_mask);


% Generate scatterer positions based on liver mask
[Y, X] = find(scaled_liver_mask);
Z = rand(size(Y)) * image_depth;
scatterer_positions = [(X - 1) * d_x - image_width / 2, zeros(size(Y)), Z];

% Modify n_scatterers to match the new scatterer_positions
n_scatterers = size(scatterer_positions, 1);

% Modify scatterer_amplitudes to match the new scatterer_positions
scatterer_amplitudes = ones(n_scatterers, 1);

% Initialize image matrix
image = zeros(n_lines, round(fs * image_depth / c));

disp(size(scatterer_positions));
% Delay-and-sum beamforming loop
for line = 1:n_lines
    % Define the line focus
    x_focus = -image_width / 2 + (line - 1) * d_x;
    focus = [x_focus, 0, image_depth / 2];

    % Set transmit and receive focus
    xdc_center_focus(Th, focus);
    xdc_focus(Th, 0, focus);
    xdc_focus_times(Th, 0, zeros(1, n_elements));

    % Simulate the received signal from the scatterers
    [rf_data, t_start] = calc_scat_multi(Th, Th, scatterer_positions, scatterer_amplitudes);

    % Delay-and-sum beamforming
    delays = zeros(1, n_elements);
    for element = 1:n_elements
        delays(element) = sqrt((focus(1) - (element - (n_elements + 1) / 2) * (element_width + kerf))^2 + focus(3)^2) / c;
    end
    delays = delays - min(delays);
    rf_data_shifted = zeros(size(rf_data));
    for element = 1:n_elements
        rf_data_shifted(:, element) = circshift(rf_data(:, element), round(delays(element) * fs));
    end

    % Store the beamformed signal in the image matrix
    sum_rf_data_shifted = sum(rf_data_shifted, 2)';
    target_length = size(image, 2);
    current_length = length(sum_rf_data_shifted);

    % Find the greatest common divisor (GCD) of the target and current lengths
    gcd_length = gcd(target_length, current_length);

    % Calculate the intermediate upsampling and downsampling factors
    upsample_factor = target_length / gcd_length;
    downsample_factor = current_length / gcd_length;

    % Perform resampling in two steps
    intermediate_resampled = resample(sum_rf_data_shifted, upsample_factor, 1);
    final_resampled = resample(intermediate_resampled, 1, downsample_factor);

    image(line, :) = final_resampled;

end


% Release the memory allocated for the transducer
xdc_free(Th);

% Envelope detection
image_env = abs(hilbert(image));

% Log-compression
dynamic_range = 60; % Dynamic range in dB
image_dB = 20 * log10(image_env / max(max(image_env)));
image_dB(image_dB < -dynamic_range) = -dynamic_range;

% Display the B-mode image
figure;
imagesc([-image_width / 2, image_width / 2] * 1e3, [0, image_depth] * 1e3, image_dB);
colormap('gray');
axis('image');
xlabel('Lateral position (mm)');
ylabel('Depth (mm)');
title('B-mode ultrasound image');
colorbar;
caxis([-dynamic_range, 0]);

% 15. Clean up (end Field II)
field_end;
