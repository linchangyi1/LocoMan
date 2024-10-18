import torch
from estimator.base_state_estimator import BaseStateEstimator


class BiManualStateEstimator(BaseStateEstimator):
    def __init__(self, robot):
        super().__init__(robot)
        # the position is aligned with the body frame, the orientation is aligned with the default world frame
        self._torso_rot_mat_w2b[:, :, :] = 0.
        self._torso_rot_mat_w2b[:, 2, 0] = 1.
        self._torso_rot_mat_w2b[:, 1, 1] = 1.
        self._torso_rot_mat_w2b[:, 0, 2] = -1.

    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        self._torso_rot_mat_w2b[env_ids, :, :] = 0.
        self._torso_rot_mat_w2b[env_ids, 2, 0] = 1.
        self._torso_rot_mat_w2b[env_ids, 1, 1] = 1.
        self._torso_rot_mat_w2b[env_ids, 0, 2] = -1.

    def update(self):
        pass

