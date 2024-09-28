import math
import numpy as np
from math import sin, cos
from enum import Enum, auto

DTYPE = np.float16

class CoordinateAxis(Enum):
    X = auto()
    Y = auto()
    Z = auto()

def coordinateRotation(axis:CoordinateAxis, theta:float) -> np.ndarray:
    """
    Compute rotation matrix for coordinate transformation. Note that
    coordinateRotation(CoordinateAxis:X, .1) * v will rotate v by -.1 radians
    this transforms into a frame rotated by .1 radians!.
    """
    s = sin(float(theta))
    c = cos(float(theta))
    R:np.ndarray = None
    if axis is CoordinateAxis.X:
        R = np.array([1, 0, 0, 0, c, s, 0, -s, c], dtype=DTYPE).reshape((3,3))
    elif axis is CoordinateAxis.Y:
        R = np.array([c, 0, -s, 0, 1, 0, s, 0, c], dtype=DTYPE).reshape((3,3))
    elif axis is CoordinateAxis.Z:
        R = np.array([c, s, 0, -s, c, 0, 0, 0, 1], dtype=DTYPE).reshape((3,3))

    return R

class Quaternion:
    def __init__(self, w:float=1, x:float=0, y:float=0, z:float=0):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self._norm = np.sqrt(self.w*self.w+self.x*self.x+self.y*self.y+self.z*self.z, dtype=DTYPE)

    def toNumpy(self):
        """convert to an (4,1) numpy array"""
        return np.array([self.w,self.x,self.y,self.z], dtype=DTYPE).reshape((4,1))
    
    def unit(self):
        """return the unit quaternion"""
        return Quaternion(self.w/self._norm,self.x/self._norm,self.y/self._norm,self.z/self._norm)

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def reverse(self):
        """return the reverse rotation representation as the same as the transpose op of rotation matrix"""
        return Quaternion(-self.w,self.x,self.y,self.z)

    def inverse(self):
        return Quaternion(self.w/(self._norm*self._norm),-self.x/(self._norm*self._norm),-self.y/(self._norm*self._norm),-self.z/(self._norm*self._norm))
    
    def __str__(self) -> str:
        return '['+str(self.w)+', '+str(self.x)+', '+str(self.y)+', '+str(self.z)+']'

# Orientation tools
def quat_product(q:Quaternion, p:Quaternion)->Quaternion:
    w = q.w*p.w - q.x*p.x - q.y*p.y - q.z*p.z
    x = q.w*p.x + q.x*p.w + q.y*p.z + q.z*p.y
    y = q.w*p.y + q.y*p.w - q.x*p.z + q.z*p.x
    z = q.w*p.z + q.z*p.w + q.x*p.y - q.y*p.x
    return Quaternion(w,x,y,z)

def rpy_to_quat(rpy)->Quaternion:
    cy = cos(rpy[2] * 0.5)
    sy = sin(rpy[2] * 0.5)
    cp = cos(rpy[1] * 0.5)
    sp = sin(rpy[1] * 0.5)
    cr = cos(rpy[0] * 0.5)
    sr = sin(rpy[0] * 0.5)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q

def get_quat_from_rpy(rpy):
    cy = cos(rpy[2] * 0.5)
    sy = sin(rpy[2] * 0.5)
    cp = cos(rpy[1] * 0.5)
    sp = sin(rpy[1] * 0.5)
    cr = cos(rpy[0] * 0.5)
    sr = sin(rpy[0] * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([x, y, z, w], dtype=DTYPE)

def get_rot_from_normals(world_normal, ground_normal):
    """
    Get rotation matrix from two plane normals
    """
    axis = np.cross(world_normal, ground_normal)
    theta = np.arccos(world_normal.dot(ground_normal))
    return axis_angle_to_rot(axis, theta)

def axis_angle_to_rot(k, theta):
    c_t = cos(theta)
    s_t = sin(theta)
    v_t = 1 - c_t

    R_axis_angle = np.array([
        k[0]*k[0]*v_t + c_t,        k[0]*k[1]*v_t - k[2]*s_t,   k[0]*k[2]*v_t + k[1]*s_t,
        k[0]*k[1]*v_t + k[2]*s_t,   k[1]*k[1]*v_t + c_t,        k[1]*k[2]*v_t - k[0]*s_t,
        k[0]*k[2]*v_t - k[1]*s_t,   k[1]*k[2]*v_t + k[0]*s_t,   k[1]*k[1]*v_t + c_t
        ], dtype=DTYPE).reshape((3,3))
    
    return R_axis_angle.T

def axis_angle_to_quat(k, theta):
    q = Quaternion()
    q.w = cos(theta/2)

    s2 = sin(theta/2)
    q.x = k[0] * s2
    q.y = k[1] * s2
    q.z = k[2] * s2
    return q

def quat_to_rpy(q:Quaternion) -> np.ndarray:
    """
    Convert a quaternion to RPY. Return
    angles in (roll, pitch, yaw).
    """
    rpy = np.zeros((3,1), dtype=DTYPE)
    as_ = np.min([-2.*(q.x*q.z-q.w*q.y),.99999])
    # roll
    rpy[0] = np.arctan2(2.*(q.y*q.z+q.w*q.x), q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z)
    # pitch
    rpy[1] = np.arcsin(as_)
    # yaw
    rpy[2] = np.arctan2(2.*(q.x*q.y+q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)
    return rpy


def quat_to_rot_mat(q):
    """
    Convert a quaternion to a rotation matrix.  This matrix represents a
    coordinate transformation into the frame which has the orientation specified
    by the quaternion
    """
    e0 = q[0]
    e1 = q[1]
    e2 = q[2]
    e3 = q[3]
    R = np.array([1 - 2 * (e2 * e2 + e3 * e3), 2 * (e1 * e2 - e0 * e3),
                  2 * (e1 * e3 + e0 * e2), 2 * (e1 * e2 + e0 * e3),
                  1 - 2 * (e1 * e1 + e3 * e3), 2 * (e2 * e3 - e0 * e1),
                  2 * (e1 * e3 - e0 * e2), 2 * (e2 * e3 + e0 * e1),
                  1 - 2 * (e1 * e1 + e2 * e2)], 
                  dtype=DTYPE).reshape((3,3))
    return R.T

def rpy_to_rot(rpy)->np.ndarray:
    """
    Convert RPY to a rotation matrix
    """
    R = coordinateRotation(CoordinateAxis.X, rpy[0]) @\
        coordinateRotation(CoordinateAxis.Y, rpy[1]) @\
        coordinateRotation(CoordinateAxis.Z, rpy[2])
    return R

def rot_to_quat(rot:np.ndarray)->Quaternion:
    """
    Convert a coordinate transformation matrix to an orientation quaternion.
    """
    q = Quaternion()
    r = rot.T.copy() # important
    tr = np.trace(r)
    if tr>0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        q.w = 0.25 * S
        q.x = (r[2,1] - r[1,2])/S
        q.y = (r[0,2] - r[2,0])/S
        q.z = (r[1,0] - r[0,1])/S

    elif (r[0, 0] > r[1, 1]) and (r[0, 0] > r[2, 2]):
        S = math.sqrt(1.0 + r[0,0] - r[1,1] - r[2,2]) * 2.0
        q.w = (r[2,1] - r[1,2])/S
        q.x = 0.25 * S
        q.y = (r[0,1] + r[1,0])/S
        q.z = (r[0,2] + r[2,0])/S

    elif r[1,1]>r[2,2]:
        S = math.sqrt(1.0 + r[1,1] -r[0,0] -r[2,2]) * 2.0
        q.w = (r[0,2] - r[2,0])/S
        q.x = (r[0,1] + r[1,0])/S
        q.y = 0.25 * S
        q.z = (r[1,2] + r[2,1])/S
        
    else:
        S = math.sqrt(1.0 + r[2,2] - r[0,0] - r[1,1]) * 2.0
        q.w = (r[1,0] - r[0,1])/S
        q.x = (r[0,2] + r[2,0])/S
        q.y = (r[1,2] + r[2,1])/S
        q.z = 0.25 * S
    
    return q

def rot_to_rpy(R:np.ndarray):
    return quat_to_rpy(rot_to_quat(R))

def deg2rad(deg:float):
    return deg*math.pi/180

def rad2deg(rad:float):
    return rad*180/math.pi

def rot_mat_to_rpy(R):
    """
    Convert a rotation matrix to Euler angles (roll, pitch, yaw) with ZYX order.
    This method is more numerically stable, especially near singularities.
    """
    sy = -R[2, 0]
    singular_threshold = 1e-6  # Threshold to check for singularity
    cy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

    # Handle singularity
    if cy < singular_threshold:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(sy, cy)
        z = 0  # Yaw is indeterminate in this case
    else:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(sy, cy)
        z = np.arctan2(R[1, 0], R[0, 0])

    return np.array([x, y, z])

def rpy_to_rot_mat(rpy):
    """
    Convert Euler angles (roll, pitch, yaw) with ZYX order to a rotation matrix.
    """
    x, y, z = rpy
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)

    R = np.array([
        [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz],
        [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
        [-sy, cy * sx, cx * cy]
    ])
    return R

def rpy_to_quaternion(rpy):
    """
    Convert Euler angles with ZYX order to a quaternion.
    """
    # Extract the Euler angles
    x = rpy[0]
    y = rpy[1]
    z = rpy[2]

    # Compute the trigonometric values
    cx = np.cos(x / 2)
    sx = np.sin(x / 2)
    cy = np.cos(y / 2)
    sy = np.sin(y / 2)
    cz = np.cos(z / 2)
    sz = np.sin(z / 2)

    # Compute the quaternion
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    w = cx * cy * cz + sx * sy * sz

    return np.array([x, y, z, w], dtype=DTYPE)

def rotation_matrix_to_euler_zyx(R):
    """
    Convert a rotation matrix to Euler angles with ZYX order.
    """
    # Check if the pitch angle is close to -90 or 90 degrees
    if np.isclose(R[2, 0], -1.0):
        # Gimbal lock at pitch = 90 degrees
        z = 0  # Set yaw to zero
        y = np.pi / 2  # Pitch is 90 degrees
        x = z + np.arctan2(R[0, 1], R[0, 2])
    elif np.isclose(R[2, 0], 1.0):
        # Gimbal lock at pitch = -90 degrees
        z = 0  # Set yaw to zero
        y = -np.pi / 2  # Pitch is -90 degrees
        x = -z + np.arctan2(-R[0, 1], -R[0, 2])
    else:
        # General case
        y = -np.arcsin(R[2, 0])  # Pitch
        cos_y = np.cos(y)
        z = np.arctan2(R[1, 0] / cos_y, R[0, 0] / cos_y)  # Yaw
        x = np.arctan2(R[2, 1] / cos_y, R[2, 2] / cos_y)  # Roll

    return np.array([x, y, z])

def euler_zyx_to_rotation_matrix(euler_zyx):
    """
    Convert Euler angles with ZYX order to a rotation matrix.
    """
    # Extract the Euler angles
    x = euler_zyx[0]
    y = euler_zyx[1]
    z = euler_zyx[2]

    # Compute the trigonometric values
    cx = np.cos(x)
    sx = np.sin(x)
    cy = np.cos(y)
    sy = np.sin(y)
    cz = np.cos(z)
    sz = np.sin(z)

    # Compute the rotation matrix
    R = np.array([
        cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
        sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
        -sy, cy * sx, cy * cx
    ], dtype=DTYPE).reshape((3,3))

    return R

def euler_zyx_to_quaternion(euler_zyx):
    """
    Convert Euler angles with ZYX order to a quaternion.
    """
    # Extract the Euler angles
    x = euler_zyx[0]
    y = euler_zyx[1]
    z = euler_zyx[2]

    # Compute the trigonometric values
    cx = np.cos(x / 2)
    sx = np.sin(x / 2)
    cy = np.cos(y / 2)
    sy = np.sin(y / 2)
    cz = np.cos(z / 2)
    sz = np.sin(z / 2)

    # Compute the quaternion
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    w = cx * cy * cz + sx * sy * sz

    return np.array([x, y, z, w], dtype=DTYPE)



from scipy.linalg import svd

def compute_transformation_matrix(points_A, points_B):
    # Ensure the points are numpy arrays
    points_A = np.array(points_A)
    points_B = np.array(points_B)
    
    # Step 1: Compute centroids
    centroid_A = np.mean(points_A, axis=0)
    centroid_B = np.mean(points_B, axis=0)
    
    # Step 2: Subtract centroids to center the points at the origin
    centered_A = points_A - centroid_A
    centered_B = points_B - centroid_B
    
    # Step 3: Compute the cross-covariance matrix
    H = centered_A.T @ centered_B
    
    # Step 4: Compute the rotation matrix using SVD
    U, _, Vt = svd(H)
    rotation_matrix = Vt.T @ U.T
    
    # Ensure a right-handed coordinate system
    if np.linalg.det(rotation_matrix) < 0:
        Vt[-1, :] *= -1
        rotation_matrix = Vt.T @ U.T
    
    # Step 5: Compute the translation vector
    translation_vector = centroid_B - rotation_matrix @ centroid_A
    
    # Step 6: Assemble the transformation matrix
    T_AB = np.eye(4)
    T_AB[:3, :3] = rotation_matrix
    T_AB[:3, 3] = translation_vector.squeeze()
    
    return T_AB





