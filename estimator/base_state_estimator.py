import torch
from utilities.rotation_utils import rot_mat_to_rpy, rot_mat_to_quaternion


class BaseStateEstimator:
    def __init__(self, robot):
        self._robot = robot
        self._num_envs = self._robot._num_envs
        self._device = self._robot._device
        self._dt = self._robot._dt

        # root state
        self._torso_pos_w = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._torso_rot_mat_w2b = torch.eye(3, dtype=torch.float, device=self._device, requires_grad=False).repeat(self._num_envs, 1, 1)
        self._torso_lin_vel_w = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._torso_ang_vel_w = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._torso_lin_acc_w = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._torso_ang_acc_w = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)


    def reset(self, env_ids: torch.Tensor):
        self._torso_pos_w[env_ids] = torch.zeros(3, dtype=torch.float, device=self._device, requires_grad=False)
        self._torso_rot_mat_w2b[env_ids] = torch.eye(3, dtype=torch.float, device=self._device, requires_grad=False)
        self._torso_lin_vel_w[env_ids] = torch.zeros(3, dtype=torch.float, device=self._device, requires_grad=False)
        self._torso_ang_vel_w[env_ids] = torch.zeros(3, dtype=torch.float, device=self._device, requires_grad=False)
        self._torso_lin_acc_w[env_ids] = torch.zeros(3, dtype=torch.float, device=self._device, requires_grad=False)
        self._torso_ang_acc_w[env_ids] = torch.zeros(3, dtype=torch.float, device=self._device, requires_grad=False)

    def update(self):
        raise NotImplementedError

    def set_robot_torso_state(self):
        self._robot._torso_pos_w[:] = self._torso_pos_w
        self._robot._torso_rot_mat_w2b[:] = self._torso_rot_mat_w2b
        self._robot._torso_rot_mat_b2w[:] = self._torso_rot_mat_w2b.transpose(-2, -1)
        self._robot._torso_rpy_w2b[:] = rot_mat_to_rpy(self._torso_rot_mat_w2b)
        self._robot._torso_quat_w2b[:] = rot_mat_to_quaternion(self._torso_rot_mat_w2b)

        self._robot._torso_lin_vel_w[:] = self._torso_lin_vel_w
        self._robot._torso_lin_vel_b[:] = torch.matmul(self._robot._torso_rot_mat_b2w, self._torso_lin_vel_w.unsqueeze(-1)).squeeze(-1)
        self._robot._torso_ang_vel_w[:] = self._torso_ang_vel_w
        self._robot._torso_ang_vel_b[:] = torch.matmul(self._robot._torso_rot_mat_b2w, self._torso_ang_vel_w.unsqueeze(-1)).squeeze(-1)

        self._robot._torso_lin_acc_w[:] = self._torso_lin_acc_w
        self._robot._torso_ang_acc_w[:] = self._torso_ang_acc_w


    @property
    def torso_pos_w(self):
        return self._torso_pos_w

    @property
    def torso_rot_mat_w2b(self):
        return self._torso_rot_mat_w2b
    
    @property
    def torso_lin_vel_w(self):
        return self._torso_lin_vel_w

    @property
    def torso_ang_vel_w(self):
        return self._torso_ang_vel_w

    @property
    def torso_rpy_w2b(self):
        return self._torso_rpy_w2b




