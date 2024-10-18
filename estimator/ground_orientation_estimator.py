"""Estimates slope of ground from IMU and foot positions."""
import numpy as np
from robot.base_robot import BaseRobot

def rotx(x):
    s = np.sin(x)
    c = np.cos(x)
    result = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return result


def roty(y):
    s = np.sin(y)
    c = np.cos(y)
    result = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return result


class GroundOrientationEstimator:
    """Estimates ground orientation from IMU reading and foot position."""
    def __init__(self, robot: BaseRobot, env_ids):
        self._robot = robot
        self._env_ids = env_ids
        self.reset()

    def reset(self):
        # self._torso_orientation_matrix_zero_yaw = self.get_zero_yaw_rotation_matrix()
        # self._foot_position_history = np.array(
        #     self._torso_orientation_matrix_zero_yaw.dot(self._robot.foot_pos_b_np[self._env_ids].T).T)
        self._foot_position_history = self._robot.foot_pos_w_np[self._env_ids]

        self._last_timestamp = self._robot.time_since_reset
        self._ground_rpy = np.zeros(3)
        # self._torso_orientation_matrix_zero_yaw = np.eye(3)
        self._ground_normal_vec = np.array([0., 0., 1.])

    def get_zero_yaw_rotation_matrix(self):
        torso_rpy = self._robot.torso_rpy_w2b_np[self._env_ids]
        return roty(torso_rpy[1]).dot(rotx(torso_rpy[0]))

    def update(self):
        new_timestamp = self._robot.time_since_reset
        dt = new_timestamp - self._last_timestamp
        self._last_timestamp = new_timestamp

        # self._torso_orientation_matrix_zero_yaw = self.get_zero_yaw_rotation_matrix()
        # torso_vel = self._torso_orientation_matrix_zero_yaw.dot(self._robot.torso_lin_vel_b_np[self._env_ids])
        torso_vel = self._robot.torso_lin_vel_w_np[self._env_ids]
        foot_contacts = self._robot.foot_contact_np[self._env_ids]
        # new_foot_positions = self._robot.foot_pos_b_np[self._env_ids]
        for idx, contact in enumerate(foot_contacts):
            if contact:
                # self._foot_position_history[idx] = self._torso_orientation_matrix_zero_yaw.dot(new_foot_positions[idx])
                self._foot_position_history[idx] = self._robot.foot_pos_w_np[self._env_ids][idx]
            else:
                self._foot_position_history[idx] -= torso_vel * dt

        # Estimate ground orientation
        ground_z = np.linalg.lstsq(self._foot_position_history, np.ones(4))[0]  # a(x-a) + b(y-b) + c(z-c) = 0; a^2 + b^2 + c^2 = 1
        ground_z /= np.linalg.norm(ground_z)
        if ground_z[2] <= 0:
            ground_z = -ground_z
        self._ground_normal_vec = ground_z

        # Convert ground normal to ground orientation
        # Here we assume the x-z plane of ground frame is aligned with the x-z
        # plane of robot frame (i.e. ground_yaw is 0)
        # robot_x = self._torso_orientation_matrix_zero_yaw[:, 0]
        robot_x = self._robot.torso_rot_mat_w2b_np[self._env_ids][:, 0]
        ground_y = np.cross(ground_z, robot_x)
        ground_y /= np.linalg.norm(ground_y)

        ground_x = np.cross(ground_y, ground_z)
        ground_x /= np.linalg.norm(ground_x)

        ground_rot_mat = np.stack((ground_x, ground_y, ground_z), axis=1)
        ground_yaw = np.arctan2(ground_rot_mat[1, 0], ground_rot_mat[0, 0])
        ground_pitch = np.arctan2(
            -ground_rot_mat[2, 0],
            np.sqrt(ground_rot_mat[2, 1]**2 + ground_rot_mat[2, 2]**2))
        ground_roll = np.arctan2(ground_rot_mat[2, 1], ground_rot_mat[2, 2])
        self._ground_rpy = np.array([ground_roll, ground_pitch, ground_yaw])  # in the zero yaw frame

    @property
    def ground_orientation_rpy(self):
        return self._ground_rpy

    @property
    def torso_orientation_matrix_zero_yaw(self):
        return self._torso_orientation_matrix_zero_yaw

    @property
    def ground_normal_vec(self):
        return self._ground_normal_vec







