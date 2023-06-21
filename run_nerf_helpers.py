import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

device = torch.device("cpu")

# Misc
def img2mse(x, y, N_rays):
    # x, y: shape: samples x 3
    # reshape to N_rays x samples, take mean across samples, return shape N_rays
    return torch.mean(((x - y) ** 2).view(N_rays, -1), dim=1)


mse2psnr = (
    lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]).to(x.get_device()))
)
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def compute_divergence_loss(
    offsets_of_inputs,
    input_points,
    point_latents,
    ray_bender,
    exact,
    chunk,
    N_rays,
    weights=None,
    backprop_into_weights=True,
):
    # offsets_of_inputs: extras["offsets"]
    # input_points: extras["initial_input_pts"]
    if exact:
        divergence_fn = divergence_exact
    else:
        divergence_fn = divergence_approx

    input_points.requires_grad = True

    def divergence_wrapper(subtensor, subtensor_latents):
        details = ray_bender(subtensor, subtensor_latents, special_loss_return=True)
        offsets = (
            details["masked_offsets"]
            if "masked_offsets" in details
            else details["unmasked_offsets"]
        )
        return divergence_fn(subtensor, offsets)

    divergence_loss = torch.cat(
        [
            divergence_wrapper(
                input_points[i : i + chunk, :], point_latents[i : i + chunk, :]
            )
            for i in range(0, input_points.shape[0], chunk)
        ],
        dim=0,
    )

    divergence_loss = torch.abs(divergence_loss)
    divergence_loss = divergence_loss ** 2
    
    if weights is not None:
        if not backprop_into_weights:
            weights = weights.detach()
        divergence_loss = weights * divergence_loss
    # don't take mean, instead reshape to N_rays x samples, take mean across samples, return shape N_rays
    return torch.mean(divergence_loss.view(N_rays, -1), dim=-1)


# from FFJORD github code
def divergence_exact(input_points, offsets_of_inputs):
    # requires three backward passes instead one like divergence_approx
    jac = _get_minibatch_jacobian(offsets_of_inputs, input_points)
    diagonal = jac.view(jac.shape[0], -1)[:, :: (jac.shape[1]+1)]
    return torch.sum(diagonal, 1)


# from FFJORD github code
def _get_minibatch_jacobian(y, x):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.
    Args:
      y: (N, ...) with a total of D_y elements in ...
      x: (N, ...) with a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)

    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(
            y[:, j],
            x,
            torch.ones_like(y[:, j], device=y.get_device()),
            retain_graph=True,
            create_graph=True,
        )[0].view(x.shape[0], -1)
        jac.append(torch.unsqueeze(dy_j_dx, 1))
    jac = torch.cat(jac, 1)
    return jac


# from FFJORD github code
def divergence_approx(input_points, offsets_of_inputs):  # , as_loss=True):
    # avoids explicitly computing the Jacobian
    e = torch.randn_like(offsets_of_inputs, device=offsets_of_inputs.get_device())
    e_dydx = torch.autograd.grad(offsets_of_inputs, input_points, e, create_graph=True)[
        0
    ]
    e_dydx_e = e_dydx * e
    approx_tr_dydx = e_dydx_e.view(offsets_of_inputs.shape[0], -1).sum(dim=1)
    return approx_tr_dydx


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 2  # Changed from 3 to 2

    embed_kwargs = {
        "include_input": True,
        "input_dims": 2,  # Changed from 3 to 2 to match the new input dimensions
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=2,
        input_ch_views=3,
        output_ch=4,
        skips=[4],
        use_viewdirs=False,
        ray_bender=None,
        ray_bending_latent_size=0,
        embeddirs_fn=None,
        num_ray_samples=None,
        approx_nonrigid_viewdirs=True,
        time_conditioned_baseline=False,
    ):
        """"""
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = False

        # nonrigid view dependence
        self.approx_nonrigid_viewdirs = approx_nonrigid_viewdirs  # approx uses finite differences, while exact uses three additional passes through ray bending in the forward pass
        self.embeddirs_fn = embeddirs_fn
        self.num_ray_samples = num_ray_samples  # netchunk needs to be divisible by both coarse and fine num_ray_samples
        
        # simple scene editing. set to None during training
        self.test_time_nonrigid_object_removal_threshold = None
        
        # naive NR-NeRF baseline
        self.time_conditioned_baseline = time_conditioned_baseline
        if self.time_conditioned_baseline:
            input_ch += ray_bending_latent_size

        # ray bending
        self.ray_bending_latent_size = ray_bending_latent_size
        self.ray_bender = (
            ray_bender,
        )  # hacky workaround to prevent ray_bender from being considered a submodule of NeRF (if it were a submodule, its parameters would show up in NeRF.parameters() and they would be added multiple times to the optimizer, once for each NeRF)

        # network layers
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)]  # note the change here
            + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)
            ]
        )


        # No need for views_linears since we're not using viewdirs
        self.views_linears = None

        # Regardless of whether use_viewdirs is True, we only need one output_linear
        self.output_linear = nn.Linear(W, output_ch)

    

    def forward(self, x):
        # Split the input into its components
        input_pts, _ = torch.split(
            x,
            [self.input_ch, x.shape[-1] - self.input_ch],
            dim=-1,
        )

        # Propagate the input points through the main network layers
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        # Run the result through the output layer to produce the final output
        outputs = self.output_linear(h)

        return outputs


        # compute Jacobian
        with torch.enable_grad():  # necessay to work properly in no_grad() mode
            jacobian = _get_minibatch_jacobian(
                bent_input_pts, initial_input_pts
            )  # shape: N x 3 x 3. N x ouptut_dims x input_dims

        # compute directional derivative: J * d
        direction = unbent_ray_direction.reshape(-1, 3, 1)  # N x 3 x 1
        directional_derivative = torch.matmul(jacobian, direction)  # N x 3 x 1

        # normalize to unit length
        directional_derivative = directional_derivative.view(-1, 3)
        normalized_directional_derivative = (
            directional_derivative
            / torch.norm(directional_derivative, dim=-1, keepdim=True)
            + 0.000001
        )

        input_views = normalized_directional_derivative.view(
            -1, 3
        )  # rays * samples x 3
        input_views = self.embeddirs_fn(input_views)  # rays * samples x input_ch_views

        return input_views


class ray_bending(nn.Module):
    def __init__(self, input_ch, ray_bending_latent_size, ray_bending_mode, embed_fn):

        super(ray_bending, self).__init__()

        self.use_positionally_encoded_input = False
        self.input_ch = input_ch if self.use_positionally_encoded_input else 3
        self.output_ch = 3  # don't change
        self.ray_bending_latent_size = ray_bending_latent_size
        self.ray_bending_mode = ray_bending_mode
        self.embed_fn = embed_fn
        self.use_rigidity_network = True
        # simple scene editing. set to None during training.
        self.rigidity_test_time_cutoff = None
        self.test_time_scaling = None

        if self.ray_bending_mode == "simple_neural":
            self.activation_function = F.relu  # F.relu, torch.sin
            self.hidden_dimensions = 64  # 32
            self.network_depth = 5  # 3 # at least 2: input -> hidden -> output
            self.skips = []  # do not include 0 and do not include depth-1
            use_last_layer_bias = False

            self.network = nn.ModuleList(
                [
                    nn.Linear(
                        self.input_ch + self.ray_bending_latent_size,
                        self.hidden_dimensions,
                    )
                ]
                + [
                    nn.Linear(
                        self.input_ch + self.hidden_dimensions, self.hidden_dimensions
                    )
                    if i + 1 in self.skips
                    else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
                    for i in range(self.network_depth - 2)
                ]
                + [
                    nn.Linear(
                        self.hidden_dimensions, self.output_ch, bias=use_last_layer_bias
                    )
                ]
            )

            # initialize weights
            with torch.no_grad():
                for i, layer in enumerate(self.network[:-1]):
                    if self.activation_function.__name__ == "sin":
                        # SIREN ( Implicit Neural Representations with Periodic Activation Functions https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
                        if type(layer) == nn.Linear:
                            a = (
                                1.0 / layer.in_features
                                if i == 0
                                else np.sqrt(6.0 / layer.in_features)
                            )
                            layer.weight.uniform_(-a, a)
                    elif self.activation_function.__name__ == "relu":
                        torch.nn.init.kaiming_uniform_(
                            layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                        )
                        torch.nn.init.zeros_(layer.bias)

                # initialize final layer to zero weights to start out with straight rays
                self.network[-1].weight.data *= 0.0
                if use_last_layer_bias:
                    self.network[-1].bias.data *= 0.0

        if self.use_rigidity_network:
            self.rigidity_activation_function = F.relu  # F.relu, torch.sin
            self.rigidity_hidden_dimensions = 32  # 32
            self.rigidity_network_depth = 3  # 3 # at least 2: input -> hidden -> output
            self.rigidity_skips = []  # do not include 0 and do not include depth-1
            use_last_layer_bias = True
            self.rigidity_tanh = nn.Tanh()

            self.rigidity_network = nn.ModuleList(
                [nn.Linear(self.input_ch, self.rigidity_hidden_dimensions)]
                + [
                    nn.Linear(
                        self.input_ch + self.rigidity_hidden_dimensions,
                        self.rigidity_hidden_dimensions,
                    )
                    if i + 1 in self.rigidity_skips
                    else nn.Linear(
                        self.rigidity_hidden_dimensions, self.rigidity_hidden_dimensions
                    )
                    for i in range(self.rigidity_network_depth - 2)
                ]
                + [
                    nn.Linear(
                        self.rigidity_hidden_dimensions, 1, bias=use_last_layer_bias
                    )
                ]
            )

            # initialize weights
            with torch.no_grad():
                for i, layer in enumerate(self.rigidity_network[:-1]):
                    if self.rigidity_activation_function.__name__ == "sin":
                        # SIREN ( Implicit Neural Representations with Periodic Activation Functions https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
                        if type(layer) == nn.Linear:
                            a = (
                                1.0 / layer.in_features
                                if i == 0
                                else np.sqrt(6.0 / layer.in_features)
                            )
                            layer.weight.uniform_(-a, a)
                    elif self.rigidity_activation_function.__name__ == "relu":
                        torch.nn.init.kaiming_uniform_(
                            layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                        )
                        torch.nn.init.zeros_(layer.bias)

                # initialize final layer to zero weights
                self.rigidity_network[-1].weight.data *= 0.0
                if use_last_layer_bias:
                    self.rigidity_network[-1].bias.data *= 0.0

    def forward(
        self, input_pts, input_latents, details=None, special_loss_return=False
    ):

        # inputs_pts: num_points x input_ch # input_ch refers to size after positional encoding
        # input_latents: num_points x ray_bending_latent size

        if special_loss_return and details is None:
            details = {}

        raw_input_pts = input_pts[
            :, :3
        ]  # positional encoding includes the raw 3D coordinates as the first three entries
        if not self.use_positionally_encoded_input:
            input_pts = raw_input_pts

        if self.ray_bending_mode == "simple_neural":
            # fully-connected network regresses offset
            h = torch.cat([input_pts, input_latents], -1)
            for i, layer in enumerate(self.network):
                h = layer(h)

                # SIREN
                if self.activation_function.__name__ == "sin" and i == 0:
                    h *= 30.0

                if (
                    i != len(self.network) - 1
                ):  # no activation function after last layer (Relu prevents backprop if the input is zero & need offsets in positive and negative directions)
                    h = self.activation_function(h)

                if i in self.skips:
                    h = torch.cat([input_pts, h], -1)

            unmasked_offsets = h
            if details is not None:
                details["unmasked_offsets"] = unmasked_offsets

        if self.use_rigidity_network:
            h = input_pts
            for i, layer in enumerate(self.rigidity_network):
                h = layer(h)

                # SIREN
                if self.rigidity_activation_function.__name__ == "sin" and i == 0:
                    h *= 30.0

                if i != len(self.rigidity_network) - 1:
                    h = self.rigidity_activation_function(h)

                if i in self.rigidity_skips:
                    h = torch.cat([input_pts, h], -1)
            rigidity_mask = (
                self.rigidity_tanh(h) + 1
            ) / 2  # close to 1 for nonrigid, close to 0 for rigid

            if self.rigidity_test_time_cutoff is not None:
                rigidity_mask[rigidity_mask <= self.rigidity_test_time_cutoff] = 0.0

        if self.use_rigidity_network:
            masked_offsets = rigidity_mask * unmasked_offsets
            if self.test_time_scaling is not None:
                masked_offsets *= self.test_time_scaling
            new_points = raw_input_pts + masked_offsets  # skip connection
            if details is not None:
                details["rigidity_mask"] = rigidity_mask
                details["masked_offsets"] = masked_offsets
        else:
            if self.test_time_scaling is not None:
                unmasked_offsets *= self.test_time_scaling
            new_points = raw_input_pts + unmasked_offsets  # skip connection

        if special_loss_return:  # used for compute_divergence_loss()
            return details
        else:  # default
            return self.embed_fn(
                new_points
            )  # apply positional encoding. num_points x input_ch


# CHANGED: Ray helpers
def get_ultrasound_rays(c2w, intrin):
    H = intrin["height"]
    W = intrin["width"]
    print(H)
    print(W)
    device = c2w.get_device()
    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))  
    i = i.t()
    j = j.t()
    c2w_rot = c2w[:3, :3] 
    pixel_coords = torch.stack([i, j, torch.zeros_like(i)], dim=0)

    # Perform batch-wise matrix multiplication
    rays_o = torch.einsum('ij,jkl->ikl', c2w_rot, pixel_coords)

    # Add translation
    rays_o += c2w[:3, -1].unsqueeze(-1).unsqueeze(-1)
    # Ray directions (a constant vector [0, 0, 1] in the camera's local space, rotated to the world space)
    rays_d = c2w[:3, :3] @ torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
    rays_d = rays_d.unsqueeze(-1).unsqueeze(-1).expand(rays_o.shape)
    return rays_o, rays_d


def get_rays_np(c2w, intrin):
    H = intrin["height"]  # This represents the height of the 2D ultrasound sweep
    W = intrin["width"]   # This represents the width of the 2D ultrasound sweep
    i, j = np.meshgrid(np.linspace(0, W-1, W), np.linspace(0, H-1, H), indexing='ij')

    # In the context of ultrasound, rays will originate from each point on the ultrasound sweep
    # The ray's origin, rays_o, is computed using the c2w matrix, which represents the 3D position and orientation of the ultrasound transducer. 
    # Instead of having a single origin point, each point on the ultrasound sweep (i,j) will have its own origin.
    rays_o = np.einsum('ij,klj->kli', c2w[:3,:3], np.stack([i, j, np.ones_like(i)], -1)) + c2w[:3,-1][None, None]

    # The ray's direction, rays_d, is the same for all rays, and it's a constant vector [0, 0, 1] in the ultrasound transducer's coordinates (assuming the sweep plane is the xy-plane). 
    # We need to rotate this direction vector from the ultrasound transducer's coordinate system to the world coordinates, using the c2w rotation matrix.
    rays_d = np.einsum('ij,k->jk', c2w[:3,:3], np.array([0, 0, 1]))[None, None].repeat(rays_o.shape[0], axis=0).repeat(rays_o.shape[1], axis=1)

    return rays_o, rays_d


def ndc_rays(intrin, near, rays_o, rays_d):
    H = intrin["height"]
    W = intrin["width"]
    focal_x = intrin["focal_x"]
    focal_y = intrin["focal_y"]

    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal_x)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal_y)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal_x)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal_y)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    device = weights.get_device()
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1], device=device), cdf], -1
    )  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0.0, 1.0, N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u).to(device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf.detach(), u, right=False)

    below = torch.max(torch.zeros_like(inds - 1, device=device), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds, device=device), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom, device=device), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def visualize_disparity_with_jet_color_scheme(depth_map_in):
    from matplotlib import cm

    color_mapping = np.array([cm.jet(i)[:3] for i in range(256)])

    max_depth = 1
    min_depth = 0
    depth_map = (
        np.clip(depth_map_in, a_max=max_depth, a_min=min_depth) / max_depth
    )  # cut off above max_depth. result is normalized to [0,1]
    depth_map = (255.0 * depth_map).astype("uint8")  # now contains int in [0,255]
    original_shape = depth_map.shape
    depth_map = color_mapping[depth_map.flatten()]
    depth_map = depth_map.reshape(original_shape + (3,))
    return depth_map


def visualize_disparity_with_blinn_phong(depth_map):
    # follows https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_reflection_model
    lightPos = np.array([1.0, 1.0, 1.0])
    lightColor = np.array([1.0, 1.0, 1.0])
    lightPower = 2.0  # 40.0
    ambientColor = np.array([0.1, 0.0, 0.0])
    diffuseColor = np.array([0.5, 0.0, 0.0])
    specColor = np.array([1.0, 1.0, 1.0])
    shininess = 2.0  # 16.0

    height, width = depth_map.shape

    # normals from depth map
    # https://stackoverflow.com/questions/53350391/surface-normal-calculation-from-depth-map-in-python
    spacing = 2.0 / (height - 1)
    zy, zx = np.gradient(depth_map, spacing)
    normal = np.dstack(
        (-zx, zy, np.ones_like(depth_map))
    )  # need to flip zy because OpenGL indexes bottom left as (0,0) (this is a guess, it simply turns out to work if zy is flipped)
    normal_length = np.linalg.norm(normal, axis=2, keepdims=True)
    normal /= normal_length  # height x width x 3

    i, j = np.meshgrid(
        np.arange(width, dtype=np.float32) / width,
        np.arange(height, dtype=np.float32) / width,
        indexing="xy",
    )  # note: if height != width then dividing the second argument by height would lead to anisotropic scaling
    vertPos = np.stack(
        [i, j, depth_map], axis=-1
    )  # height x width x 3. note that (x,y) and (depth) have different scaling factors and offsets because we don't do proper unprojection - might need to adjust them

    lightDir = -vertPos + lightPos.reshape(1, 1, 3)  # height x width x 3
    distance = np.linalg.norm(lightDir, axis=2, keepdims=True)  # height x width x 1
    lightDir /= distance
    # distance = distance ** 2
    distance = (distance + 1.0) ** 2

    def dot_product(A, B):
        return np.sum(A * B, axis=-1)

    lightDir_x_normal = dot_product(lightDir, normal)
    lambertian = np.clip(lightDir_x_normal, a_max=None, a_min=0.0).reshape(
        height, width, 1
    )  # height x width x 1

    invalid_mask = lambertian <= 0.0

    def normalize(image):
        return image / np.linalg.norm(image, axis=-1, keepdims=True)

    viewDir = normalize(-vertPos)  # height x width x 3
    halfDir = normalize(lightDir + viewDir)  # height x width x 3
    # specAngle = np.clip(dot_product(halfDir, normal), a_max=None, a_min=0.).reshape(height, width, 1) # height x width x 1
    specAngle = np.clip(dot_product(halfDir, -normal), a_max=None, a_min=0.0).reshape(
        height, width, 1
    )  # height x width x 1
    specular = specAngle ** shininess

    specular[invalid_mask] = 0.0

    colorLinear = (
        lambertian
        * diffuseColor.reshape(1, 1, 3)
        * lightColor.reshape(1, 1, 3)
        * lightPower
        / distance
        + specular
        * specColor.reshape(1, 1, 3)
        * lightColor.reshape(1, 1, 3)
        * lightPower
        / distance
        + ambientColor.reshape(1, 1, 3)
    )  # height x width x 3
    return colorLinear


def visualize_ray_bending(
    initial_input_pts, input_pts, filename_prefix, subsampled_target=None
):
    # initial_input_pts: rays x samples_per_ray x 3
    # input_pts: rays x samples_per_ray x 3

    if subsampled_target is None:
        subsampled_target = 100

    if (
        len(input_pts.shape) == 4
    ):  # height x width x samples_per_ray x 3 -- happens in render() after batchify_rays() returns
        input_pts = input_pts.reshape(-1, input_pts.shape[-2], 3)
        initial_input_pts = initial_input_pts.reshape(
            -1, initial_input_pts.shape[-2], 3
        )
    num_rays, samples_per_ray, _ = input_pts.shape

    if subsampled_target < num_rays:
        indices = np.random.choice(num_rays, size=[subsampled_target], replace=False)
    else:
        indices = np.arange(num_rays)

    def _ray_mesh(input_pts):
        rays_string = ""
        num_lines = 0
        for ray_samples in input_pts[indices]:
            # ray_samples: samples_per_ray x 3
            for i in range(samples_per_ray - 1):
                num_lines += 1
                start_x, start_y, start_z = ray_samples[i]
                end_x, end_y, end_z = ray_samples[i + 1]
                eps = 0.00001
                rays_string += (
                    "v "
                    + str(start_x)
                    + " "
                    + str(start_y)
                    + " "
                    + str(start_z)
                    + "\n"
                    + "v "
                    + str(start_x + eps)
                    + " "
                    + str(start_y + eps)
                    + " "
                    + str(start_z + eps)
                    + "\n"
                    + "v "
                    + str(end_x)
                    + " "
                    + str(end_y)
                    + " "
                    + str(end_z)
                    + "\n"
                )
        for i in range(num_lines):
            # faces are 1-indexed
            first_vertex_index = i * 3 + 1
            rays_string += (
                "f "
                + str(first_vertex_index)
                + " "
                + str(first_vertex_index + 1)
                + " "
                + str(first_vertex_index + 2)
                + "\n"
            )
        return rays_string

    with open(filename_prefix + "_bent.obj", "w") as file:
        file.write(_ray_mesh(input_pts))
    with open(filename_prefix + "_not_bent.obj", "w") as file:
        file.write(_ray_mesh(initial_input_pts))

    def _delta_mesh(start_pts, end_pts):
        delta_string = ""
        start_pts = start_pts.reshape(-1, 3)
        end_pts = end_pts.reshape(-1, 3)
        for (start_x, start_y, start_z), (end_x, end_y, end_z) in zip(
            start_pts, end_pts
        ):
            eps = 0.00001
            delta_string += (
                "v "
                + str(start_x)
                + " "
                + str(start_y)
                + " "
                + str(start_z)
                + "\n"
                + "v "
                + str(start_x + eps)
                + " "
                + str(start_y + eps)
                + " "
                + str(start_z + eps)
                + "\n"
                + "v "
                + str(end_x)
                + " "
                + str(end_y)
                + " "
                + str(end_z)
                + "\n"
            )
        for i in range(len(start_pts)):
            # faces are 1-indexed
            first_vertex_index = i * 3 + 1
            delta_string += (
                "f "
                + str(first_vertex_index)
                + " "
                + str(first_vertex_index + 1)
                + " "
                + str(first_vertex_index + 2)
                + "\n"
            )
        return delta_string

    with open(filename_prefix + "_deltas.obj", "w") as file:
        file.write(_delta_mesh(initial_input_pts[indices], input_pts[indices]))


def determine_nerf_volume_extent(
    render_function, poses, intrinsics, render_kwargs, args
):
    # the nerf volume has some extent, but this extent is not fixed. this function computes (somewhat approximate) minimum and maximum coordinates along each axis. it considers all cameras (their positions and point samples along the rays of their corners).
    poses = torch.Tensor(poses).cuda()
    print("poses shape", poses.shape)
    critical_rays_o = []
    critical_rays_d = []
    for c2w, intrin in zip(poses, intrinsics): 
        print(c2w.shape)
        print(c2w)
        this_c2w = c2w[:3, :4]
        rays_o, rays_d = get_ultrasound_rays(this_c2w, intrin)
        camera_corners_o = torch.stack(
            [rays_o[0, 0, :], rays_o[-1, 0, :], rays_o[0, -1, :], rays_o[-1, -1, :]]
        )  # 4x3
        camera_corners_d = torch.stack(
            [rays_d[0, 0, :], rays_d[-1, 0, :], rays_d[0, -1, :], rays_d[-1, -1, :]]
        )  # 4x3
        critical_rays_o.append(camera_corners_o)
        critical_rays_d.append(camera_corners_d)
    critical_rays_o = torch.cat(critical_rays_o, dim=0)
    critical_rays_d = torch.cat(critical_rays_d, dim=0)  # N x 3

    num_rays = critical_rays_o.shape[0]

    additional_pixel_information = {
        "ray_invalidity": torch.zeros(num_rays),
        "rgb_validity": torch.ones(num_rays),
        "ray_bending_latents": torch.zeros(
            (num_rays, intrinsics[0]["ray_bending_latent_size"])
        ),
    }

    with torch.no_grad():
        rgb, disp, acc, details_and_rest = render_function(
            critical_rays_o,
            critical_rays_d,
            chunk=128,
            detailed_output=True,
            additional_pixel_information=additional_pixel_information,
            **render_kwargs
        )

    critical_ray_points = details_and_rest["initial_input_pts"].reshape(-1, 3)  # N x 3
    camera_positions = poses[:, :3, 3]  # N x 3

    output_camera_visualization = True
    if output_camera_visualization:
        output_folder = os.path.join(args.rootdir, args.expname, "logs/")
        with open(os.path.join(output_folder, "cameras.obj"), "w") as mesh_file:
            beginning = (
                details_and_rest["initial_input_pts"][:, 0, :].detach().cpu().numpy()
            )
            end = details_and_rest["initial_input_pts"][:, -1, :].detach().cpu().numpy()
            for x, y, z in beginning:
                mesh_file.write(
                    "v " + str(x) + " " + str(y) + " " + str(z) + " 0.0 1.0 0.0\n"
                )
            for x, y, z in end:
                mesh_file.write(
                    "v " + str(x) + " " + str(y) + " " + str(z) + " 1.0 0.0 0.0\n"
                )
            for x, y, z in end:
                mesh_file.write(
                    "v "
                    + str(x + 0.00001)
                    + " "
                    + str(y)
                    + " "
                    + str(z)
                    + " 1.0 0.0 0.0\n"
                )
            for x, y, z in camera_positions.detach().cpu().numpy():
                mesh_file.write(
                    "v " + str(x) + " " + str(y) + " " + str(z) + " 0.0 0.0 1.0\n"
                )
            for x, y, z in camera_positions.detach().cpu().numpy():
                mesh_file.write(
                    "v "
                    + str(x + 0.00001)
                    + " "
                    + str(y)
                    + " "
                    + str(z)
                    + " 0.0 0.0 1.0\n"
                )
            for x, y, z in camera_positions.detach().cpu().numpy():
                mesh_file.write(
                    "v "
                    + str(x)
                    + " "
                    + str(y + 0.00001)
                    + " "
                    + str(z)
                    + " 0.0 0.0 1.0\n"
                )
            num_vertices = beginning.shape[0]
            for i in range(num_vertices):
                i += 1
                mesh_file.write(
                    "f "
                    + str(i)
                    + " "
                    + str(i + num_vertices)
                    + " "
                    + str(i + 2 * num_vertices)
                    + "\n"
                )
            offset = 3 * num_vertices
            num_cameras = camera_positions.detach().cpu().numpy().shape[0]
            for i in range(num_cameras):
                i += 1
                mesh_file.write(
                    "f "
                    + str(offset + i)
                    + " "
                    + str(offset + i + num_cameras)
                    + " "
                    + str(offset + i + 2 * num_cameras)
                    + "\n"
                )

    critical_points = torch.cat([critical_ray_points, camera_positions], dim=0)
    min_point = torch.min(critical_points, dim=0)[0]
    max_point = torch.max(critical_points, dim=0)[0]

    # add some extra space around the volume. stretch away from the center of the volume.
    center = (min_point + max_point) / 2.0
    min_point -= center
    max_point -= center
    min_point *= 1.1
    max_point *= 1.1
    min_point += center
    max_point += center

    return min_point, max_point