import torch
import imageio
import numpy as np
import os
from train import ( 
    render_path,
    create_nerf,
    get_parallelized_render_function
)
from load_llff import load_llff_data

def config_parser():

    import configargparse

    parser = configargparse.ArgumentParser()
    code_folder = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument(
        "--config",
        is_config_file=True,
        help="config file path",
        default=os.path.join(code_folder, "configs", "default.txt"),
    )
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument("--datadir", type=str, help="input data directory")
    parser.add_argument(
        "--rootdir",
        type=str,
        help="root folder where experiment results will be stored: rootdir/expname/",
    )

    # training options
    parser.add_argument("--netdepth", type=int, default=8, help="layers in network")
    parser.add_argument("--netwidth", type=int, default=256, help="channels per layer")
    parser.add_argument(
        "--netdepth_fine", type=int, default=8, help="layers in fine network"
    )
    parser.add_argument(
        "--netwidth_fine",
        type=int,
        default=256,
        help="channels per layer in fine network",
    )
    parser.add_argument(
        "--N_iters", type=int, default=200000, help="number of training iterations"
    )
    parser.add_argument(
        "--N_rand",
        type=int,
        default=32 * 32 * 4,
        help="batch size (number of random rays per gradient step)",
    )
    parser.add_argument("--lrate", type=float, default=5e-4, help="learning rate")
    parser.add_argument(
        "--lrate_decay",
        type=int,
        default=250000,
        help="exponential learning rate decay",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=1024 * 32,
        help="number of rays processed in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--netchunk",
        type=int,
        default=1024 * 64,
        help="number of pts sent through network in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--no_reload", action="store_true", help="do not reload weights from saved ckpt"
    )
    parser.add_argument(
        "--ft_path",
        type=str,
        default=None,
        help="specific weights npy file to reload for coarse network",
    )
    parser.add_argument("--seed", type=int, default=-1, help="seeding numpy")
    parser.add_argument(
        "--ray_bending",
        type=str,
        default="None",
        help="which type of ray bending to use (None or simple_neural)",
    )
    parser.add_argument(
        "--ray_bending_latent_size",
        type=int,
        default=32,
        help="size of per-frame autodecoding latent vector used for ray bending",
    )
    parser.add_argument(
        "--approx_nonrigid_viewdirs",
        action="store_true",
        help="approximate nonrigid view directions of the bent ray instead of exact",
    )
    parser.add_argument(
        "--time_conditioned_baseline",
        action="store_true",
        help="use the naive NR-NeRF baseline described in the paper",
    )

    parser.add_argument(
        "--train_block_size",
        type=int,
        default=0,
        help="number of consecutive timesteps to use for training",
    )
    parser.add_argument(
        "--test_block_size",
        type=int,
        default=0,
        help="number of consecutive timesteps to use for testing",
    )

    # rendering options
    parser.add_argument(
        "--N_samples", type=int, default=64, help="number of coarse samples per ray"
    )
    parser.add_argument(
        "--N_importance",
        type=int,
        default=0,
        help="number of additional fine samples per ray",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=1.0,
        help="set to 0. for no jitter, 1. for jitter",
    )
    parser.add_argument(
        "--offsets_loss_weight",
        type=float,
        default=0.0,
        help="set to 0. for no offsets loss",
    )
    parser.add_argument(
        "--divergence_loss_weight",
        type=float,
        default=0.0,
        help="set to 0. for no divergence loss",
    )
    parser.add_argument(
        "--rigidity_loss_weight",
        type=float,
        default=0.0,
        help="set to 0. for no rigidity loss",
    )
    parser.add_argument(
        "--use_viewdirs", action="store_true", help="use full 5D input instead of 3D"
    )
    parser.add_argument(
        "--i_embed",
        type=int,
        default=0,
        help="set 0 for default positional encoding, -1 for none",
    )
    parser.add_argument(
        "--multires",
        type=int,
        default=10,
        help="log2 of max freq for positional encoding (3D location)",
    )
    parser.add_argument(
        "--multires_views",
        type=int,
        default=4,
        help="log2 of max freq for positional encoding (2D direction)",
    )
    parser.add_argument(
        "--raw_noise_std",
        type=float,
        default=0.0,
        help="std dev of noise added to regularize sigma_a output, 1e0 recommended",
    )
    parser.add_argument(
        "--render_factor",
        type=int,
        default=0,
        help="downsampling factor to speed up rendering, set 4 or 8 for fast preview",
    )
    parser.add_argument(
        "--render_test",
        action="store_true",
        help="render the test set instead of render_poses path",
    )

    # training options
    parser.add_argument(
        "--precrop_iters",
        type=int,
        default=0,
        help="number of steps to train on central crops",
    )
    parser.add_argument(
        "--precrop_frac",
        type=float,
        default=0.5,
        help="fraction of img taken for central crops",
    )
    parser.add_argument("--debug", action="store_true", help="enable checking for NaNs")

    # dataset options
    parser.add_argument(
        "--dataset_type", type=str, default="llff", help="options: llff"
    )

    # llff flags
    parser.add_argument(
        "--factor", type=int, default=8, help="downsample factor for LLFF images"
    )
    parser.add_argument(
        "--spherify", action="store_true", help="set for spherical 360 scenes"
    )
    parser.add_argument(
        "--bd_factor",
        type=str,
        default="0.75",
        help="scales the overall scene, NeRF uses 0.75. is ignored.",
    )

    # logging/saving options
    parser.add_argument(
        "--i_print",
        type=int,
        default=100,
        help="frequency of console printout and metric loggin",
    )
    parser.add_argument(
        "--i_img", type=int, default=500, help="frequency of tensorboard image logging"
    )
    parser.add_argument(
        "--i_weights", type=int, default=1000, help="frequency of weight ckpt saving"
    )
    parser.add_argument(
        "--i_testset", type=int, default=50000, help="frequency of testset saving"
    )
    parser.add_argument(
        "--i_video",
        type=int,
        default=50000,
        help="frequency of render_poses video saving",
    )

    return parser



def load_checkpoint_and_render_images(checkpoint_path, output_dir, args):
    # TODO: change that back to cuda
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))


    # Set up rendering parameters and camera poses
    images, poses, bds, render_poses, i_test = load_llff_data(
                args.datadir,
                factor=args.factor,
                recenter=True,
                bd_factor=args.bd_factor,
                spherify=args.spherify,
            )
    intrinsics = checkpoint['intrinsics']
    ray_bending_latents_list = checkpoint['ray_bending_latent_codes']
    dataset_extras = checkpoint['dataset_extras']

    render_kwargs_train, render_kwargs_test, _, _, _ = create_nerf(args, autodecoder_variables=ray_bending_latents_list, ignore_optimizer=True)

    coarse_model = render_kwargs_train["network_fn"]
    fine_model = render_kwargs_train["network_fine"]
    ray_bender = render_kwargs_train["ray_bender"]

    coarse_model.load_state_dict(checkpoint['network_fn_state_dict'])
    if fine_model is not None:
        fine_model.load_state_dict(checkpoint['network_fine_state_dict'])
    if ray_bender is not None:
        ray_bender.load_state_dict(checkpoint['ray_bender_state_dict'])

    parallel_render = get_parallelized_render_function(
        coarse_model=coarse_model, fine_model=fine_model, ray_bender=ray_bender
    )
    # Add this code block after loading dataset_extras

    i_test = np.arange(len(ray_bending_latents_list))

    print("Length of ray_bending_latents:", len(ray_bending_latents_list))
    print("i_test:", i_test)


    # Use the rendering function to generate images
    with torch.no_grad():
        rendering_latents = [
            ray_bending_latents_list[dataset_extras["imageid_to_timestepid"][i]]
            for i in i_test
        ]
        rgbs, disps = render_path(
            render_poses,
            [intrinsics[0] for _ in range(len(render_poses))],
            args.chunk,
            render_kwargs_test,
            ray_bending_latents=rendering_latents,
            parallelized_render_function=parallel_render,
        )
    def to8b(img):
        return (255 * np.clip(img, 0, 1)).astype(np.uint8)
    # Save the generated images
    for i, (rgb, disp) in enumerate(zip(rgbs, disps)):
        imageio.imwrite(f'{output_dir}/output_image_rgb_{i}.png', to8b(rgb))
        imageio.imwrite(f'{output_dir}/output_image_disp_{i}.png', to8b(disp / np.max(disp)))


if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()
    checkpoint_path = "results/latest.tar"
    output_dir = "results"
    load_checkpoint_and_render_images(checkpoint_path, output_dir, args)
