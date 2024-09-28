import torch
from utilities.rotation_utils import quat_to_rot_mat
from estimator.sim_base_state_estimator import SimBaseStateEstimator


class SimManStateEstimator(SimBaseStateEstimator):
    def __init__(self, robot):
        super().__init__(robot)

    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        # For single-arm manipulation, in the sim frame, the world frame is the same as the body(torso) frame at the reset frame.
        self._world_pos_sim[env_ids] = self._robot._torso_pos_sim[env_ids].clone()
        self._world_quat_sim2w[env_ids] = self._robot._torso_quat_sim2b[env_ids].clone()
        self._world_rot_mat_w2sim[env_ids] = quat_to_rot_mat(self._world_quat_sim2w[env_ids]).transpose(-2, -1)

    def update(self, env_ids: torch.Tensor=None):
        env_ids = env_ids if env_ids is not None else torch.arange(self._num_envs, device=self._device)
        self._torso_pos_w[env_ids] = torch.matmul(self._world_rot_mat_w2sim[env_ids], (self._robot._torso_pos_sim[env_ids] - self._world_pos_sim[env_ids]).unsqueeze(-1)).squeeze(-1)
        self._torso_rot_mat_w2b[env_ids] =  torch.matmul(self._world_rot_mat_w2sim[env_ids], quat_to_rot_mat(self._robot._torso_quat_sim2b[env_ids]))
        self._torso_lin_vel_w[env_ids] = torch.matmul(self._world_rot_mat_w2sim[env_ids], self._robot._torso_lin_vel_sim[env_ids].unsqueeze(-1)).squeeze(-1)
        self._torso_ang_vel_w[env_ids] = torch.matmul(self._world_rot_mat_w2sim[env_ids], self._robot._torso_ang_vel_sim[env_ids].unsqueeze(-1)).squeeze(-1)

