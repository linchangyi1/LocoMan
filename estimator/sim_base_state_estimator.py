import torch
from estimator.base_state_estimator import BaseStateEstimator
from utilities.rotation_utils import rot_mat_to_rpy, quat_to_rot_mat


class SimBaseStateEstimator(BaseStateEstimator):
    def __init__(self, robot):
        super().__init__(robot)

        # In simulation, there are three frames: sim, world, and body. The world frame has different definitions for different operation modes.
        self._world_pos_sim = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._world_quat_sim2w = torch.zeros((self._num_envs, 4), dtype=torch.float, device=self._device, requires_grad=False)
        self._world_quat_sim2w[:, 3] = 1.0
        self._world_rot_mat_w2sim = torch.eye(3, dtype=torch.float, device=self._device, requires_grad=False).repeat(self._num_envs, 1, 1)
        self._reshaped_jacobian_sim = torch.zeros_like(self._robot._jacobian_w[:, :, 0:6, 0:6]).reshape(self._num_envs, -1, 4, 3, 3)

    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        self._reshaped_jacobian_sim[env_ids] = self._robot._jacobian_w[:, :, 0:6, 0:6].reshape(self._num_envs, -1, 4, 3, 3)[env_ids].clone()

    def update(self):
        raise NotImplementedError

    def compute_jacobian_w(self):
        # for the dofs from the floating base, we also need to transfer the space(inputs) from the body frame to the world frame:
        # v^{w} = R_w^s * v^{s} = R_w^s * J^{s} * dq^{s} = R_w^s * J^{s} * R_s^w * dq^{w} = R_w^s * J^{s} * R_s^w * R_w^b * dq^{b}
        self._reshaped_jacobian_sim[:, :, 0, 0:3, 0:3] = self._robot._jacobian_sim[:, :, 0:3, 0:3]
        self._reshaped_jacobian_sim[:, :, 1, 0:3, 0:3] = self._robot._jacobian_sim[:, :, 0:3, 3:6]
        self._reshaped_jacobian_sim[:, :, 2, 0:3, 0:3] = self._robot._jacobian_sim[:, :, 3:6, 0:3]
        self._reshaped_jacobian_sim[:, :, 3, 0:3, 0:3] = self._robot._jacobian_sim[:, :, 3:6, 3:6]
        temp_jacobian_w = torch.matmul(
                                    torch.matmul(self._world_rot_mat_w2sim.reshape(-1, 1, 1, 3, 3),
                                         torch.matmul(self._reshaped_jacobian_sim,self._world_rot_mat_w2sim.reshape(-1, 1, 1, 3, 3).transpose(-2, -1))), 
                                                            self._torso_rot_mat_w2b.reshape(-1, 1, 1, 3, 3))
        self._robot._jacobian_w[:, :, 0:3, 0:3] = temp_jacobian_w[:, :, 0, 0:3, 0:3]
        self._robot._jacobian_w[:, :, 0:3, 3:6] = temp_jacobian_w[:, :, 1, 0:3, 0:3]
        self._robot._jacobian_w[:, :, 3:6, 0:3] = temp_jacobian_w[:, :, 2, 0:3, 0:3]
        self._robot._jacobian_w[:, :, 3:6, 3:6] = temp_jacobian_w[:, :, 3, 0:3, 0:3]

        # for the dofs in the joint space, only need to transfer the outpus from the sim frame to the world frame:
        # v^{w} = R_w^s * v^{s} = R_w^s * J^{s} * dq^{j}
        self._robot._jacobian_w[:, :, 0:3, 6:] = torch.matmul(self._world_rot_mat_w2sim.unsqueeze(1), self._robot._jacobian_sim[:, :, 0:3, 6:])
        self._robot._jacobian_w[:, :, 3:6, 6:] = torch.matmul(self._world_rot_mat_w2sim.unsqueeze(1), self._robot._jacobian_sim[:, :, 3:6, 6:])


    def set_foot_global_state(self):
        self._robot._foot_pos_w[:] = torch.matmul(self._world_rot_mat_w2sim, (self._robot._foot_pos_sim - self._world_pos_sim.unsqueeze(-2)).transpose(-2, -1)).transpose(-2, -1)
        self._robot._foot_vel_w[:] = torch.matmul(self._world_rot_mat_w2sim, self._robot._foot_vel_sim.transpose(-2, -1)).transpose(-2, -1)

        if self._robot._use_gripper:
            self._robot._eef_pos_w[:] = torch.matmul(self._world_rot_mat_w2sim, (self._robot._eef_pos_sim - self._world_pos_sim.unsqueeze(-2)).transpose(-2, -1)).transpose(-2, -1)
            self._robot._eef_rot_w[:] = torch.matmul(self._world_rot_mat_w2sim.unsqueeze(1), quat_to_rot_mat(self._robot._eef_quat_sim.reshape(-1, 4)).reshape(self._num_envs, -1, 3, 3))
            self._robot._eef_rpy_w[:] = rot_mat_to_rpy(self._robot._eef_rot_w.reshape(-1, 3, 3)).reshape(self._num_envs, -1, 3)



