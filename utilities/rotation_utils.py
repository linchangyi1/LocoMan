import torch
import numpy as np


@torch.jit.script
def rot_mat_to_rpy(R):
    # Gimbal lock detection
    gimbal_lock_upper = torch.isclose(R[:, 2, 0], torch.tensor(1.0, device=R.device))
    gimbal_lock_lower = torch.isclose(R[:, 2, 0], torch.tensor(-1.0, device=R.device))
    
    # Pitch (y)
    y = -torch.asin(R[:, 2, 0])  # Asin is in [-pi/2, pi/2]
    
    # For cases not in gimbal lock, calculate roll (x) and yaw (z)
    x = torch.where(gimbal_lock_upper | gimbal_lock_lower, torch.tensor(0.0, device=R.device),
                    torch.atan2(R[:, 2, 1], R[:, 2, 2]))
    z = torch.where(gimbal_lock_upper | gimbal_lock_lower, torch.atan2(R[:, 1, 0], R[:, 0, 0]),
                    torch.atan2(R[:, 1, 0], R[:, 0, 0]))
    
    # Adjust for gimbal lock upper
    z[gimbal_lock_upper] = torch.atan2(-R[gimbal_lock_upper, 0, 1], R[gimbal_lock_upper, 1, 1])
    # Adjust for gimbal lock lower
    z[gimbal_lock_lower] = torch.atan2(-R[gimbal_lock_lower, 0, 1], R[gimbal_lock_lower, 1, 1])

    y[gimbal_lock_upper] = torch.pi / 2
    y[gimbal_lock_lower] = -torch.pi / 2

    # Correct angle wrapping for x, y, z to [-pi, pi] range
    x = (x + torch.pi) % (2 * torch.pi) - torch.pi
    y = (y + torch.pi) % (2 * torch.pi) - torch.pi
    z = (z + torch.pi) % (2 * torch.pi) - torch.pi

    rpy = torch.stack([x, y, z], dim=1)

    return rpy


@torch.jit.script
def rpy_to_rot_mat(rpy):
    x = rpy[:, 0]
    y = rpy[:, 1]
    z = rpy[:, 2]

    cx = torch.cos(x)
    sx = torch.sin(x)
    cy = torch.cos(y)
    sy = torch.sin(y)
    cz = torch.cos(z)
    sz = torch.sin(z)

    R = torch.zeros(rpy.shape[0], 3, 3, dtype=rpy.dtype, device=rpy.device)
    R[:, 0, 0] = cy * cz
    R[:, 0, 1] = sx * sy * cz - cx * sz
    R[:, 0, 2] = cx * sy * cz + sx * sz
    R[:, 1, 0] = cy * sz
    R[:, 1, 1] = sx * sy * sz + cx * cz
    R[:, 1, 2] = cx * sy * sz - sx * cz
    R[:, 2, 0] = -sy
    R[:, 2, 1] = sx * cy
    R[:, 2, 2] = cx * cy

    return R

@torch.jit.script
def rpy_to_quaternion(rpy):
    x = rpy[:, 0] / 2
    y = rpy[:, 1] / 2
    z = rpy[:, 2] / 2

    cx = torch.cos(x)
    sx = torch.sin(x)
    cy = torch.cos(y)
    sy = torch.sin(y)
    cz = torch.cos(z)
    sz = torch.sin(z)

    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz
    qw = cx * cy * cz + sx * sy * sz

    return torch.stack([qx, qy, qz, qw], dim=1)

@torch.jit.script
def quat_to_rot_mat(quat):
    # Normalize the quaternion
    norm_quat = quat / torch.norm(quat, dim=1, keepdim=True)

    # Extract the components of the normalized quaternion
    x, y, z, w = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    # Precompute squared components
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # Compute the rotation matrix components
    R = torch.zeros(quat.shape[0], 3, 3, dtype=quat.dtype, device=quat.device)
    R[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[:, 0, 1] = 2.0 * (xy - wz)
    R[:, 0, 2] = 2.0 * (xz + wy)
    R[:, 1, 0] = 2.0 * (xy + wz)
    R[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[:, 1, 2] = 2.0 * (yz - wx)
    R[:, 2, 0] = 2.0 * (xz - wy)
    R[:, 2, 1] = 2.0 * (yz + wx)
    R[:, 2, 2] = 1.0 - 2.0 * (xx + yy)

    return R

@torch.jit.script
def rot_mat_to_quaternion(R):
    # Allocate memory for the output quaternions
    quat = torch.zeros(R.shape[0], 4, dtype=R.dtype, device=R.device)
    
    # Compute the trace of the matrix
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    # Check the value of the trace to determine which formula to use
    for i in range(R.shape[0]):
        if tr[i] > 0:
            S = torch.sqrt(tr[i] + 1.0) * 2 # S=4*qw 
            quat[i, 3] = 0.25 * S
            quat[i, 0] = (R[i, 2, 1] - R[i, 1, 2]) / S
            quat[i, 1] = (R[i, 0, 2] - R[i, 2, 0]) / S 
            quat[i, 2] = (R[i, 1, 0] - R[i, 0, 1]) / S 
        elif (R[i, 0, 0] > R[i, 1, 1]) and (R[i, 0, 0] > R[i, 2, 2]):
            S = torch.sqrt(1.0 + R[i, 0, 0] - R[i, 1, 1] - R[i, 2, 2]) * 2 # S=4*qx
            quat[i, 3] = (R[i, 2, 1] - R[i, 1, 2]) / S
            quat[i, 0] = 0.25 * S
            quat[i, 1] = (R[i, 0, 1] + R[i, 1, 0]) / S 
            quat[i, 2] = (R[i, 0, 2] + R[i, 2, 0]) / S 
        elif R[i, 1, 1] > R[i, 2, 2]:
            S = torch.sqrt(1.0 + R[i, 1, 1] - R[i, 0, 0] - R[i, 2, 2]) * 2 # S=4*qy
            quat[i, 3] = (R[i, 0, 2] - R[i, 2, 0]) / S
            quat[i, 0] = (R[i, 0, 1] + R[i, 1, 0]) / S 
            quat[i, 1] = 0.25 * S
            quat[i, 2] = (R[i, 1, 2] + R[i, 2, 1]) / S
        else:
            S = torch.sqrt(1.0 + R[i, 2, 2] - R[i, 0, 0] - R[i, 1, 1]) * 2 # S=4*qz
            quat[i, 3] = (R[i, 1, 0] - R[i, 0, 1]) / S
            quat[i, 0] = (R[i, 0, 2] + R[i, 2, 0]) / S
            quat[i, 1] = (R[i, 1, 2] + R[i, 2, 1]) / S
            quat[i, 2] = 0.25 * S

    return quat

@torch.jit.script
def copysign(a, b):
    # type: (float, torch.Tensor) -> torch.Tensor
  a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
  return torch.abs(a) * torch.sign(b)


@torch.jit.script
def get_euler_xyz_from_quaternion(q):
  qx, qy, qz, qw = 0, 1, 2, 3
  # roll (x-axis rotation)
  sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
  cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
      q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
  roll = torch.atan2(sinr_cosp, cosr_cosp)

  # pitch (y-axis rotation)
  sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
  pitch = torch.where(
      torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

  # yaw (z-axis rotation)
  siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
  cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
      q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
  yaw = torch.atan2(siny_cosp, cosy_cosp)

  return torch.stack(
      (roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)), dim=1)

@torch.jit.script
def rpy_vel_to_skew_synmetric_mat(x):
    """
    Converts a batch of vectors (N, 3) to a batch of skew-symmetric matrices (N, 3, 3).
    """
    # Extracting components of x
    zeros = torch.zeros_like(x[:, 0])
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]

    # Constructing the skew-symmetric matrices
    skew_symmetric_matrices = torch.stack([
        torch.stack([zeros, -x3, x2], dim=1),
        torch.stack([x3, zeros, -x1], dim=1),
        torch.stack([-x2, x1, zeros], dim=1)
    ], dim=1)

    return skew_symmetric_matrices

@torch.jit.script
def skew_symmetric_mat_to_rpy_vel(M):
    """
    Converts a batch of skew-symmetric matrices (N, 3, 3) to a batch of vectors (N, 3).
    """
    # Extracting components of M
    m12, m13, m21, m23, m31, m32 = M[:, 0, 1], M[:, 0, 2], M[:, 1, 0], M[:, 1, 2], M[:, 2, 0], M[:, 2, 1]

    # Constructing the vectors
    x = torch.stack([m32, m13, m21], dim=1) / 2

    return x






