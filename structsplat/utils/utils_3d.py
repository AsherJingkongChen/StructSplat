import math
from einops import rearrange
import torch
from structsplat.vggt.utils.rotation import quat_to_mat
import torch.nn.functional as F

def getProjectionMatrix(znear, zfar, fovX, fovY):
    """
    Reference: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/graphics_utils.py
    """
    dtype = fovX.dtype
    device = fovX.device
    tanHalfFovY = torch.tan(fovY / 2)
    tanHalfFovX = torch.tan(fovX / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, dtype=dtype, device=device)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def depth_to_world_coords_points(
    depth_map,
    extrinsic,
    intrinsic
):
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map: Gaussian Depth map of shape (B, S, G, H, W).
        intrinsic: Camera intrinsic matrix of shape (B, S, 3, 3).
        extrinsic: Camera extrinsic matrix of shape (B, S, 3, 4) or (B, S, 4, 4). OpenCV camera coordinate convention, world to camera.

    Returns:
        World coordinates (B, S, H, W, 3)
        Camera coordinates (B, S, H, W, 3)
    """
    if depth_map is None:
        return None, None, None

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic) # (B S G H W 3)

    # Get R and T
    R, T = extrinsic[..., :3, :3], extrinsic[..., :3, 3] # (B S 3 3), (B S 3) 
    T = rearrange(T, "b s c -> b s 1 1 1 c") # (B S 1 1 1 3)

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = torch.einsum("bsrc,bsghwr->bsghwc", R, cam_coords_points - T)

    return world_coords_points, cam_coords_points


def depth_to_cam_coords_points(depth_map, intrinsic):
    H, W = depth_map.shape[-2:]
    device = depth_map.device
    dtype = depth_map.dtype
    assert intrinsic.shape[-2:] == (3, 3)
    assert torch.all(intrinsic[..., 0, 1] == 0).item() and torch.all(intrinsic[..., 1, 0] == 0).item()

    # Intrinsic parameters
    fu, fv = intrinsic[..., 0, 0], intrinsic[..., 1, 1]
    cu, cv = intrinsic[..., 0, 2], intrinsic[..., 1, 2]
    fu = rearrange(fu, "b s -> b s 1 1 1")
    fv = rearrange(fv, "b s -> b s 1 1 1")
    cu = rearrange(cu, "b s -> b s 1 1 1")
    cv = rearrange(cv, "b s -> b s 1 1 1")

    # Generate grid of pixel coordinates
    u, v = torch.meshgrid(
        torch.arange(W, dtype=dtype, device=device),
        torch.arange(H, dtype=dtype, device=device),
        indexing='xy'
    )
    u = rearrange(u, "h w -> 1 1 1 h w")
    v = rearrange(v, "h w -> 1 1 1 h w")

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = torch.stack((x_cam, y_cam, z_cam), axis=-1)

    return cam_coords # (B S G H W 3)


def get_c2w(w2c):
    # Validate shapes
    if w2c.shape[-2:] != (4, 4) and w2c.shape[-2:] != (3, 4):
        raise ValueError(f"w2c must be of shape (...,4,4) or (...,3,4), got {w2c.shape}.")
    
    B, S = w2c.shape[:2]

    # Extract R and T if not provided
    R = w2c[..., :3, :3]  # (B,S,3,3)
    T = w2c[..., :3, 3:]  # (B,S,3,1)

    # Transpose R
    R_c2w = R.transpose(-2, -1)  # (B,S,3,3)
    T_c2w = - (R_c2w @ T) # (B,S,3,1)

    c2w = torch.zeros([B, S, 4, 4], dtype=w2c.dtype, device=w2c.device)
    c2w[..., :3, :3] = R_c2w
    c2w[..., :3, 3:] = T_c2w
    c2w[..., 3, 3] = 1.0

    return c2w


def compute_rays(c2w, fxfycxcy, h=None, w=None):
    """
    Args:
        c2w (torch.tensor): [b, v, 4, 4]
        fxfycxcy (torch.tensor): [b, v, 4]
        h (int): height of the image
        w (int): width of the image
    Returns:
        ray_o (torch.tensor): [b, v, 3, h, w]
        ray_d (torch.tensor): [b, v, 3, h, w]
    """
    device = c2w.device
    dtype = c2w.dtype
    b, v = c2w.size()[:2]
    c2w = c2w.reshape(b * v, 4, 4)

    fx, fy, cx, cy = fxfycxcy[:,:, 0], fxfycxcy[:,:,  1], fxfycxcy[:,:,  2], fxfycxcy[:,:,  3]
    h_orig = int(2 * cy.max().item())  # Original height (estimated from the intrinsic matrix)
    w_orig = int(2 * cx.max().item())  # Original width (estimated from the intrinsic matrix)
    if h is None or w is None:
        h, w = h_orig, w_orig

    # in case the ray/image map has different resolution than the original image
    if h_orig != h or w_orig != w:
        fx = fx * w / w_orig
        fy = fy * h / h_orig
        cx = cx * w / w_orig
        cy = cy * h / h_orig

    fxfycxcy = fxfycxcy.reshape(b * v, 4)
    y, x = torch.meshgrid(torch.arange(h, dtype=dtype, device=device), torch.arange(w, dtype=dtype, device=device), indexing="ij")
    # y, x = y.to(device), x.to(device)
    x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
    y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
    x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b*v, h*w, 3]
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b*v, h*w, 3]

    ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
    ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)

    return ray_o, ray_d


def get_posed_input(images=None, ray_o=None, ray_d=None, method="default_plucker"):
    '''
    Args:
        images: [b, v, c, h, w]
        ray_o: [b, v, 3, h, w]
        ray_d: [b, v, 3, h, w]
        method: Method for creating pose conditioning
    Returns:
        posed_images: [b, v, c+6, h, w] or [b, v, 6, h, w] if images is None
    '''

    if method == "custom_plucker":
        o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
        nearest_pts = ray_o + o_dot_d * ray_d
        pose_cond = torch.cat([ray_d, nearest_pts], dim=2)
        
    elif method == "aug_plucker":
        o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
        nearest_pts = ray_o + o_dot_d * ray_d
        o_cross_d = torch.cross(ray_o, ray_d, dim=2)
        pose_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=2)
        
    else:  # default_plucker
        o_cross_d = torch.cross(ray_o, ray_d, dim=2)
        pose_cond = torch.cat([o_cross_d, ray_d], dim=2)

    if images is None:
        return pose_cond
    else:
        return torch.cat([images * 2.0 - 1.0, pose_cond], dim=2)
    

def pose_encoding_to_extri(
    pose_encoding,
    pose_encoding_type="absT_quaR_FoV",
):
    if pose_encoding_type == "absT_quaR_FoV":
        # T = pose_encoding[..., :3]
        # quat = pose_encoding[..., 3:7]
        # R = quat_to_mat(quat)
        # extrinsics = torch.cat([R, T[..., None]], dim=-1)

        extrinsics = torch.zeros(pose_encoding.shape[:2] + (4, 4), dtype=pose_encoding.dtype, device=pose_encoding.device)
        extrinsics[..., :3, :3] = quat_to_mat(pose_encoding[..., 3:7]) # R
        extrinsics[..., :3, 3] = pose_encoding[..., :3] # T
        extrinsics[..., 3, 3] = 1.0
    else:
        raise NotImplementedError
    return extrinsics  # Shape: (B, S, 4, 4)


def pose_encoding_to_intri(
    pose_encoding,
    image_size_hw,  # e.g., (256, 512)
    pose_encoding_type="absT_quaR_FoV",
):  
    if pose_encoding_type == "absT_quaR_FoV":
        fov_h = pose_encoding[..., 7]
        fov_w = pose_encoding[..., 8]
        H, W = image_size_hw
        fy = (H / 2.0) / torch.tan(fov_h / 2.0)
        fx = (W / 2.0) / torch.tan(fov_w / 2.0)
        intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3),  dtype=pose_encoding.dtype, device=pose_encoding.device)
        intrinsics[..., 0, 0] = fx
        intrinsics[..., 1, 1] = fy
        intrinsics[..., 0, 2] = W / 2
        intrinsics[..., 1, 2] = H / 2
        intrinsics[..., 2, 2] = 1.0  # Set the homogeneous coordinate to 1
    else:
        raise NotImplementedError
    return intrinsics  # Shape: (B, S, 3, 3)


def rotation_6d_to_matrix(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def quat_wxyz_to_mat(q):
        # Convert WXYZ to XYZW
        q_xyzw = q[...,(1,2,3,0)]
        return quat_to_mat(q_xyzw)