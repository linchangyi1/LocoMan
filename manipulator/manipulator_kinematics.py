import math
import numpy as np
np.set_printoptions(precision=4, suppress=True)
from scipy.spatial.transform import Rotation
import torch
import math


class ManipulatorKinematics:
    def __init__(self, cfg):
        self.leg_sign = [-1, 1]

        self.T_f_1 = np.array([np.eye(4), np.eye(4)])
        self.T_1_2 = np.array([np.eye(4), np.eye(4)])
        self.T_2_e = np.array([np.eye(4), np.eye(4)])
        self.T_f_e = np.array([np.eye(4), np.eye(4)])

        for i in range(2):
            self.T_f_1[i, 0:3, 3] = np.array([0.017662, 0.033 * self.leg_sign[i], -0.182+0.213])
            self.T_1_2[i, 0:3, 3] = np.array([0, 0, -0.0531])
            self.T_2_e[i, 0:3, 3] = np.array([0, 0, -0.086])

        self.joint_angles = np.zeros((2, 3))

    def forward_kinematics(self, joint_angles, idx=[], degrees=False):
        rescale = 1.0
        if degrees:
            rescale = math.pi / 180
        self.joint_angles[idx] = np.array(joint_angles).reshape(len(idx), 3) * rescale
        for i in idx:
            self.T_f_1[i, 0:3, 0:3] = Rotation.from_euler('y', self.joint_angles[i, 0], degrees=False).as_matrix()
            self.T_1_2[i, 0:3, 0:3] = Rotation.from_euler('x', self.joint_angles[i, 1], degrees=False).as_matrix()
            self.T_2_e[i, 0:3, 0:3] = Rotation.from_euler('z', self.joint_angles[i, 2], degrees=False).as_matrix()
        self.T_f_e[idx] = self.T_f_1[idx] @ self.T_1_2[idx] @ self.T_2_e[idx]
        return self.T_f_e[idx, 0:3, 0:3]


class ManipulatorKinematicsTorch:
    def __init__(self, cfg, device):
        self.device = device
        self.leg_sign = torch.tensor([-1, 1], device=self.device)

        self.T_f_1 = torch.eye(4, device=self.device).unsqueeze(0).repeat(2, 1, 1)
        self.T_1_2 = torch.eye(4, device=self.device).unsqueeze(0).repeat(2, 1, 1)
        self.T_2_e = torch.eye(4, device=self.device).unsqueeze(0).repeat(2, 1, 1)
        self.T_f_e = torch.eye(4, device=self.device).unsqueeze(0).repeat(2, 1, 1)

        for i in range(2):
            self.T_f_1[i, 0:3, 3] = torch.tensor([0.017662, 0.033 * self.leg_sign[i], -0.182+0.213], device=self.device)
            self.T_1_2[i, 0:3, 3] = torch.tensor([0, 0, -0.0531], device=self.device)
            self.T_2_e[i, 0:3, 3] = torch.tensor([0, 0, -0.086], device=self.device)


    def euler_to_rot_matrix(self, angle, axis):
        """Generate rotation matrix for a single Euler angle."""
        c, s = torch.cos(angle), torch.sin(angle)
        if axis == 'x':
            return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], device=self.device)
        elif axis == 'y':
            return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device=self.device)
        elif axis == 'z':
            return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], device=self.device)

    def forward_kinematics(self, joint_angles, degrees=False):
        rescale = torch.tensor(math.pi / 180 if degrees else 1.0, device=self.device)
        joint_angles = joint_angles.reshape((2, 3)) * rescale
        for i in range(2):
            self.T_f_1[i, 0:3, 0:3] = self.euler_to_rot_matrix(joint_angles[i, 0], 'y')
            self.T_1_2[i, 0:3, 0:3] = self.euler_to_rot_matrix(joint_angles[i, 1], 'x')
            self.T_2_e[i, 0:3, 0:3] = self.euler_to_rot_matrix(joint_angles[i, 2], 'z')
        self.T_f_e[:] = torch.matmul(torch.matmul(self.T_f_1, self.T_1_2), self.T_2_e)
        return self.T_f_e



if __name__ == '__main__':
    manipulator_kinematics = ManipulatorKinematics()
    manipulator_kinematics.forward_kinematics([math.pi, 0, 0], idx=[0])



