import torch
from utilities.rotation_utils import quat_to_rot_mat
from estimator.sim_base_state_estimator import SimBaseStateEstimator


class SimBiManualStateEstimator(SimBaseStateEstimator):
    def __init__(self, robot):
        super().__init__(robot)

    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        # For bimanual manipulation, in the sim frame, the world frame always has the same position as the body frame,
        # and its orientation is defined as the body frame rotated 90 degrees around the body frame's y-axis for intuitive control.
        # This means that currently we can't control the torso because it's relatively fixed in the world frame.
        self._torso_rot_mat_w2b[env_ids, :, :] = 0.
        self._torso_rot_mat_w2b[:, [0, 1, 2], [2, 1, 0]] = torch.tensor([-1., 1., 1.], device=self._device)
        self.update(env_ids)

    def update(self, env_ids: torch.Tensor=None):
        env_ids = env_ids if env_ids is not None else torch.arange(self._num_envs, device=self._device)
        self._world_pos_sim[env_ids] = self._robot._torso_pos_sim[env_ids].clone()
        self._world_quat_sim2w[env_ids] = self._robot._torso_quat_sim2b[env_ids].clone()
        self._world_rot_mat_w2sim[env_ids] = torch.matmul(self._torso_rot_mat_w2b[env_ids], quat_to_rot_mat(self._world_quat_sim2w[env_ids]).transpose(-2, -1))
