import torch
from utilities.rotation_utils import rpy_to_rot_mat, rot_mat_to_rpy, quat_to_rot_mat
from estimator.sim_base_state_estimator import SimBaseStateEstimator


class SimLocoStateEstimator(SimBaseStateEstimator):
    def __init__(self, robot):
        super().__init__(robot)
        self._torso_rpy_w2b = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._world_rpy_sim = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)

    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        self.update(env_ids)

    def update(self, env_ids: torch.Tensor=None):
        env_ids = env_ids if env_ids is not None else torch.arange(self._num_envs, device=self._device)
        # For locomotion, in the sim frame, the world frame always has the same xy values and yaw angle as the body frame,
        # and its z value and roll/pitch angles are always zero.
        # This means that we can control the x/y/yaw velocities, z position, and roll/pitch angles of the torso in the world frame.
        self._world_pos_sim[env_ids, 0:2] = self._robot._torso_pos_sim[env_ids, 0:2]  # x, y
        torso_rpy_sim = rot_mat_to_rpy(quat_to_rot_mat(self._robot._torso_quat_sim2b))
        self._world_rpy_sim[env_ids, 2] = torso_rpy_sim[env_ids, 2]  # yaw
        self._world_rot_mat_w2sim[env_ids] = rpy_to_rot_mat(self._world_rpy_sim[env_ids]).transpose(-2, -1)

        # update robot base state in the world frame
        self._torso_pos_w[env_ids, 2] = self._robot._torso_pos_sim[env_ids, 2]
        self._torso_rpy_w2b[env_ids, 0:2] = torso_rpy_sim[env_ids, 0:2]
        self._torso_rot_mat_w2b[env_ids] = rpy_to_rot_mat(self._torso_rpy_w2b[env_ids])
        self._torso_lin_vel_w[env_ids] = torch.matmul(self._world_rot_mat_w2sim[env_ids], self._robot._torso_lin_vel_sim[env_ids].unsqueeze(-1)).squeeze(-1)
        self._torso_ang_vel_w[env_ids] = torch.matmul(self._world_rot_mat_w2sim[env_ids], self._robot._torso_ang_vel_sim[env_ids].unsqueeze(-1)).squeeze(-1)

