
import torch


def quaternion_multiply(q1, q2):
    '''
        Multiply two quaternions.
        The quaternions are represented as tensors of shape (..., 4),
        where the last dimension represents (w, x, y, z).
        The multiplication is defined as:
        q1 * q2 = (w1*w2 - x1*x2 - y1*y2 - z1*z2,
                    w1*x2 + x1*w2 + y1*z2 - z1*y2,
                    w1*y2 - x1*z2 + y1*w2 + z1*x2,
                    w1*z2 + x1*y2 - y1*x2 + z1*w2)
        where q1 = (w1, x1, y1, z1) and
        q2 = (w2, x2, y2, z2).
        The result is a tensor of the same shape as the input quaternions.
        Args:
            q1 (torch.Tensor): First quaternion tensor of shape (..., 4).
            q2 (torch.Tensor): Second quaternion tensor of shape (..., 4).
        Returns:
            torch.Tensor: Resulting quaternion tensor of shape (..., 4).        
    '''
    w1, x1, y1, z1, = q1.unbind(-1)
    w2, x2, y2, z2, = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)

def quaternion_conjugate(q):
    '''
        Compute the conjugate of a quaternion.
        The conjugate of a quaternion q = (w, x, y, z) is
        q* = (w, -x, -y, -z).
    '''
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)

def xyzw2wxzy(q):
    return q[...,(3,0,1,2)]

def ensure_positive_hemisphere_quaternion(q):
    """
    Ensure all quaternions are in the positive hemisphere by checking the sign
    of the real part (w component) and flipping if necessary.
    
    Args:
        quaternions: Tensor of quaternions with shape (..., 4) in the order (w, x, y, z)
        
    Returns:
        Tensor of quaternions in the positive hemisphere with the same shape
    """
    # Create a mask for quaternions with negative real part
    negative_mask = q[..., 0] < 0
    zero_mask = q[..., 0] == 0

    for i in range(1, 4):
        negative_mask = torch.logical_or(negative_mask, torch.logical_and(zero_mask, q[..., i] < 0))
        zero_mask = torch.logical_and(zero_mask, q[..., i] == 0)

    q[negative_mask] = -q[negative_mask]  # Flip the sign of quaternions in the negative hemisphere
    return q