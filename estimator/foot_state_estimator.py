import torch
from estimator.base_state_estimator import BaseStateEstimator
from utilities.rotation_utils import rot_mat_to_rpy
import time
from utilities.orientation_utils_numpy import compute_transformation_matrix


class FootStateEstimator(BaseStateEstimator):
    def __init__(self, robot):
        super().__init__(robot)
        self._origin_foot_pos_b = torch.zeros_like(self._robot.foot_pos_b, device=self._device, requires_grad=False)
        self._last_T = torch.eye(4, device=self._device, requires_grad=False).repeat(self._num_envs, 1, 1)
        self._current_T = torch.eye(4, device=self._device, requires_grad=False).repeat(self._num_envs, 1, 1)
        self._delta_R = torch.eye(3, device=self._device, requires_grad=False).repeat(self._num_envs, 1, 1)
        self._last_timestamp = time.time()

    def reset(self, env_ids: torch.Tensor):
        self._last_timestamp = time.time()
        super().reset(env_ids)
        self._origin_foot_pos_b[env_ids] = self._robot.foot_pos_b[env_ids].clone()
        self._last_T[env_ids] = torch.eye(4, device=self._device, requires_grad=False)
        self._current_T[env_ids] = torch.eye(4, device=self._device, requires_grad=False)
        self._delta_R[env_ids] = torch.eye(3, device=self._device, requires_grad=False)

    def update(self):
        self._dt = time.time() - self._last_timestamp
        self._last_timestamp = time.time()
        self._last_T = self._current_T.clone()
        # self._update_current_T_torch()
        self._update_current_T_numpy()

        self._torso_pos_w[:] = self._current_T[:, :3, 3]
        last_torso_lin_vel_w = self._torso_lin_vel_w.clone()
        self._torso_lin_vel_w[:] = (self._current_T[:, :3, 3] - self._last_T[:, :3, 3]) / self._dt
        self._torso_lin_acc_w[:] = (self._torso_lin_vel_w - last_torso_lin_vel_w) / self._dt

        self._torso_rot_mat_w2b[:] = self._current_T[:, :3, :3]
        last_torso_ang_vel_w = self._torso_ang_vel_w.clone()

        self._delta_R[:] = torch.matmul(self._current_T[:, :3, :3], self._last_T[:, :3, :3].transpose(-2, -1))
        delta_R_w = torch.matmul(torch.matmul(self._last_T[:, :3, :3], self._delta_R), self._last_T[:, :3, :3].transpose(-2, -1))
        self._torso_ang_vel_w[:] = rot_mat_to_rpy(delta_R_w) / self._dt
        self._torso_ang_acc_w[:] = (self._torso_ang_vel_w - last_torso_ang_vel_w) / self._dt

    def _update_current_T_torch(self):
        expanded_contact = self._robot._desired_foot_contact.unsqueeze(-1).expand(-1, -1, 3)
        centroid_A = torch.mean(self._robot.foot_pos_b[expanded_contact].view(self._num_envs, -1, 3), dim=1, keepdim=True)
        centroid_B = torch.mean(self._origin_foot_pos_b[expanded_contact].view(self._num_envs, -1, 3), dim=1, keepdim=True)
        centered_A = self._robot.foot_pos_b[expanded_contact].view(self._num_envs, -1, 3) - centroid_A
        centered_B = self._origin_foot_pos_b[expanded_contact].view(self._num_envs, -1, 3) - centroid_B

        H = torch.matmul(centered_A.transpose(-2, -1), centered_B)
        U, _, Vt = torch.linalg.svd(H)
        
        # Adjust Vt based on the determinant to ensure a right-handed coordinate system
        det = torch.det(torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1)))
        if det < 0:
            Vt[..., -1] *= -1  # Negate the last column of Vt

        rotation_matrix = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))
        translation_vector = centroid_B - torch.matmul(rotation_matrix, centroid_A.transpose(-2, -1)).transpose(-2, -1)
        self._current_T[:, :3, :3] = rotation_matrix
        self._current_T[:, :3, 3] = translation_vector.squeeze(-2)

    def _update_current_T_numpy(self):
        for i in range(self._num_envs):
            contact_state = self._robot.desired_foot_contact_np[i]
            T_np = compute_transformation_matrix(self._robot.foot_pos_b_np[i, contact_state, ::], self._robot.origin_foot_pos_b_np[i, contact_state, ::])
            self._current_T[i, :3, :3] = torch.tensor(T_np[:3, :3], dtype=torch.float, device=self._device, requires_grad=False)
            self._current_T[i, :3, 3] = torch.tensor(T_np[:3, 3], dtype=torch.float, device=self._device, requires_grad=False)

        
