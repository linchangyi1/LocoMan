from config.config import Cfg
from estimator.state_estimator import StateEstimator
from fsm.finite_state_machine import FSM_State, SingleLegIndex, FSM_Situation
import torch
from utilities.rotation_utils import rot_mat_to_rpy


class BaseRobot:
    def __init__(self, cfg: Cfg):
        # general settings
        self._cfg = cfg
        self._num_envs = self._cfg.sim.num_envs
        self._device = self._cfg.sim.sim_device
        self._use_gripper = self._cfg.sim.use_gripper
        self._dt = self._cfg.motor_control.dt
        self._num_step = torch.zeros(self._num_envs, dtype=torch.long, device=self._device, requires_grad=False)

        # logging settings
        self._log_info = self._cfg.logging.log_info
        self._log_interval = self._cfg.logging.log_interval
        self._log_info_now = False

        # rigid body names
        self._torso_names = self._cfg.asset.torso_names
        self._hip_names = self._cfg.asset.hip_names
        self._thigh_names = self._cfg.asset.thigh_names
        self._calf_names = self._cfg.asset.calf_names
        self._foot_names = self._cfg.asset.foot_names
        self._num_legs = len(self._foot_names)
        
        # foot indices within the four legs
        self._foot_idx_of_four = torch.arange(self._num_legs, dtype=torch.long, device=self._device, requires_grad=False)

        # root state
        self._torso_pos_w = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._torso_quat_w2b = torch.zeros((self._num_envs, 4), dtype=torch.float, device=self._device, requires_grad=False)
        self._torso_rot_mat_w2b = torch.zeros((self._num_envs, 3, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._torso_rot_mat_b2w = torch.zeros((self._num_envs, 3, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._torso_rpy_w2b = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._torso_lin_vel_b = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._torso_lin_vel_w = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._torso_ang_vel_b = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._torso_ang_vel_w = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._torso_lin_acc_b = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._torso_lin_acc_w = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._torso_ang_acc_b = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._torso_ang_acc_w = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)


        # joint state
        self._motors = self._cfg.asset.motors
        self._num_joints = self._motors._num_motors
        self._dog_num_joints = 12
        # print('num_joints: ', self._num_joints)
        self._joint_pos = torch.zeros((self._num_envs, self._num_joints), dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_vel = torch.zeros((self._num_envs, self._num_joints), dtype=torch.float, device=self._device, requires_grad=False)

        # robot jacobian in world frame
        floating_base = self._cfg.sim.use_real_robot or (not self._cfg.get_asset_config().asset_options.fix_base_link)
        num_all_links = 1 * int(floating_base) + 4*self._num_legs + 6 * int(self._use_gripper)
        num_all_joints = self._num_joints + 6 * int(floating_base)
        self._jacobian_w = torch.zeros((self._num_envs, num_all_links, 6, num_all_joints), dtype=torch.float, device=self._device, requires_grad=False)

        # foot state
        self._foot_contact = torch.ones((self._num_envs, self._num_legs), dtype=torch.bool, device=self._device, requires_grad=False)
        self._desired_foot_contact = torch.ones((self._num_envs, self._num_legs), dtype=torch.bool, device=self._device, requires_grad=False)
        self._foot_pos_hip = torch.zeros((self._num_envs, self._num_legs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._foot_vel_hip = torch.zeros((self._num_envs, self._num_legs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._foot_pos_b = torch.zeros((self._num_envs, self._num_legs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._foot_vel_b = torch.zeros((self._num_envs, self._num_legs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._foot_rot_b = torch.eye(3, dtype=torch.float, device=self._device, requires_grad=False).repeat(self._num_envs, self._num_legs, 1, 1)
        self._foot_pos_w = torch.zeros((self._num_envs, self._num_legs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._foot_vel_w = torch.zeros((self._num_envs, self._num_legs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._foot_rot_w = torch.eye(3, dtype=torch.float, device=self._device, requires_grad=False).repeat(self._num_envs, self._num_legs, 1, 1)
        self._foot_T_w = torch.eye(4, dtype=torch.float, device=self._device, requires_grad=False).repeat(self._num_envs, self._num_legs, 1, 1)
        self._origin_foot_pos_b = torch.zeros((self._num_envs, self._num_legs, 3), dtype=torch.float, device=self._device, requires_grad=False)

        # foot jacobian
        self._foot_jacobian_b = torch.zeros((self._num_envs, self._num_legs, 3, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._foot_jacobian_w = torch.zeros((self._num_envs, self._num_legs, 3, 6+self._num_joints), dtype=torch.float, device=self._device, requires_grad=False)
        
        # robot features
        self._HIP_OFFSETS = torch.tensor(self._cfg.asset.hip_offsets, dtype=torch.float, device=self._device, requires_grad=False)
        self._l_hip = self._cfg.asset.l_hip
        self._l_thi = self._cfg.asset.l_thi
        self._l_cal = self._cfg.asset.l_cal
        self.l_hip_sign = (-1)**(self._foot_idx_of_four + 1)

        # joint indices
        self._dog_joint_idx = torch.tensor(self._cfg.asset.dog_joint_idx, dtype=torch.long, device=self._device, requires_grad=False)
        self._manipulator_joint_idx = torch.tensor(self._cfg.asset.manipulator_joint_idx, dtype=torch.long, device=self._device, requires_grad=False)

        if self._use_gripper:
            self._eef_names = self._cfg.asset.eef_names
            self._eef_pos_hip = torch.zeros((self._num_envs, len(self._eef_names), 3), dtype=torch.float, device=self._device, requires_grad=False)
            self._eef_vel_hip = torch.zeros((self._num_envs, len(self._eef_names), 3), dtype=torch.float, device=self._device, requires_grad=False)
            self._eef_pos_b = torch.zeros((self._num_envs, len(self._eef_names), 3), dtype=torch.float, device=self._device, requires_grad=False)
            self._eef_vel_b = torch.zeros((self._num_envs, len(self._eef_names), 3), dtype=torch.float, device=self._device, requires_grad=False)
            self._eef_pos_w = torch.zeros((self._num_envs, len(self._eef_names), 3), dtype=torch.float, device=self._device, requires_grad=False)
            self._eef_vel_w = torch.zeros((self._num_envs, len(self._eef_names), 3), dtype=torch.float, device=self._device, requires_grad=False)
            self._eef_jacobian_b = torch.zeros((self._num_envs, len(self._eef_names), 3, 6), dtype=torch.float, device=self._device, requires_grad=False)
            self._eef_jacobian_w = torch.zeros((self._num_envs, len(self._eef_names), 3, 6+self._num_joints), dtype=torch.float, device=self._device, requires_grad=False)
            self._eef_rpy_w = torch.zeros((self._num_envs, len(self._eef_names), 3), dtype=torch.float, device=self._device, requires_grad=False)
            self._eef_rot_w = torch.eye(3, dtype=torch.float, device=self._device, requires_grad=False).repeat(self._num_envs, len(self._eef_names), 1, 1)
            self._eef_T_foot = torch.eye(4, dtype=torch.float, device=self._device, requires_grad=False).repeat(self._num_envs, len(self._eef_names), 1, 1)
            self._eef_T_w = torch.eye(4, dtype=torch.float, device=self._device, requires_grad=False).repeat(self._num_envs, len(self._eef_names), 1, 1)
            self._close_gripper = torch.ones((self._num_envs, len(self._eef_names)), dtype=torch.bool, device=self._device, requires_grad=False)
            self._gripper_angles = torch.zeros((self._num_envs, len(self._eef_names)), dtype=torch.float, device=self._device, requires_grad=False)
            self._gripper_desired_angles = torch.zeros((self._num_envs, len(self._eef_names)), dtype=torch.float, device=self._device, requires_grad=False)
            from manipulator.manipulator_kinematics import ManipulatorKinematicsTorch
            self._manipulator_kinematics = ManipulatorKinematicsTorch( self._cfg, self._device)

        # state estimator
        self._state_estimator = StateEstimator(self, self._cfg.sim.use_real_robot)

        # finite state machine
        self._pre_fsm_state = FSM_State.STANCE
        self._cur_fsm_state = FSM_State.STANCE
        self._nex_fsm_state = FSM_State.STANCE
        self._pre_single_leg_idx = SingleLegIndex.RIGHT
        self._cur_single_leg_idx = None
        self._nex_single_leg_idx = None
        self._fsm_situation = FSM_Situation.NORMAL

    def reset(self):
        raise NotImplementedError
    
    def step(self):
        raise NotImplementedError

    def _apply_action(self):
        raise NotImplementedError

    def _update_state(self, reset_estimator=False, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, dtype=torch.long, device=self._device, requires_grad=False)
        self._update_sensors()
        if self._cfg.sim.use_real_robot:
            self._update_foot_local_state()
            self._update_torso_state(reset_estimator, env_ids)  # this may use foot local state
            self._update_foot_global_state()  # this will use base state
        else:
            self._update_torso_state(reset_estimator, env_ids)
            self._update_foot_global_state()  # this will use base state
            self._update_foot_local_state()  # this will use foot global state
        if self._use_gripper:
            self._update_eef_state()
        if reset_estimator:
            self._origin_foot_pos_b[env_ids] = self._foot_pos_b[env_ids].clone()

    def _update_sensors(self):
        raise NotImplementedError

    def _update_foot_local_state(self):
        self._update_foot_contact_state()
        self._update_foot_jocabian_position_velocity()
    
    def _update_foot_contact_state(self):
        raise NotImplementedError
    
    def _update_foot_jocabian_position_velocity(self):
        raise NotImplementedError

    def _update_torso_state(self, reset_estimator=False, env_ids=None):
        if reset_estimator:
            self._state_estimator[self._cur_fsm_state].reset(env_ids)
        else:
            self._state_estimator[self._cur_fsm_state].update()
        self._state_estimator[self._cur_fsm_state].set_robot_torso_state()

    def _update_foot_global_state(self):
        raise NotImplementedError

    def _update_eef_state(self):
        dog_joints = self._joint_pos[:, self._dog_joint_idx].reshape(self._num_envs, self._num_legs, -1).clone()
        self._foot_rot_b[:, :, 0, 0] = torch.cos(dog_joints[:, :, 1]+dog_joints[:, :, 2])
        self._foot_rot_b[:, :, 0, 1] = 0.0
        self._foot_rot_b[:, :, 0, 2] = torch.sin(dog_joints[:, :, 1]+dog_joints[:, :, 2])
        self._foot_rot_b[:, :, 1, 0] = torch.sin(dog_joints[:, :, 0]) * torch.sin(dog_joints[:, :, 1]+dog_joints[:, :, 2])
        self._foot_rot_b[:, :, 1, 1] = torch.cos(dog_joints[:, :, 0])
        self._foot_rot_b[:, :, 1, 2] = -torch.sin(dog_joints[:, :, 0]) * torch.cos(dog_joints[:, :, 1]+dog_joints[:, :, 2])
        self._foot_rot_b[:, :, 2, 0] = -torch.cos(dog_joints[:, :, 0]) * torch.sin(dog_joints[:, :, 1]+dog_joints[:, :, 2])
        self._foot_rot_b[:, :, 2, 1] = torch.sin(dog_joints[:, :, 0])
        self._foot_rot_b[:, :, 2, 2] = torch.cos(dog_joints[:, :, 0]) * torch.cos(dog_joints[:, :, 1]+dog_joints[:, :, 2])
        self._foot_T_w[:, :, 0:3, 0:3] = torch.matmul(self._torso_rot_mat_w2b.unsqueeze(1), self._foot_rot_b)
        self._foot_T_w[:, :, 0:3, 3] = self._foot_pos_w.clone()
        for i in range(self._num_envs):
            self._eef_T_foot[i, :] = self._manipulator_kinematics.forward_kinematics(self._joint_pos[i, self._manipulator_joint_idx])
        self._eef_T_w[:] = torch.matmul(self._foot_T_w[:, 0:len(self._eef_names)], self._eef_T_foot)
        self._eef_pos_w[:] = self._eef_T_w[:, :, 0:3, 3]
        self._eef_rot_w[:] = self._eef_T_w[:, :, 0:3, 0:3]
        self._eef_rpy_w[:] = rot_mat_to_rpy(self._eef_rot_w.reshape(-1, 3, 3)).reshape(self._num_envs, -1, 3)

    def set_desired_foot_contact(self, desired_foot_contact):
        self._desired_foot_contact[:] = desired_foot_contact

    def set_gripper_action(self, close_gripper):
        self._close_gripper[:] = close_gripper

    def set_gripper_angles(self, gripper_angles):
        self._gripper_desired_angles[:] = gripper_angles

    def _compute_foot_jacobian(self, desired_joint_pos=None):
        dy = self._l_hip * self.l_hip_sign
        dz1 = -self._l_thi
        dz2 = -self._l_cal

        joint_pos = self._joint_pos.clone() if desired_joint_pos is None else desired_joint_pos.clone()
        joint_pos = torch.cat((joint_pos[:, 0:3], joint_pos[:, 6:9], joint_pos[:, 12:]), dim=1) if self._use_gripper else joint_pos
        joint_pos = joint_pos.view(-1, 4, 3)

        s1 = torch.sin(joint_pos[:, :, 0])
        s2 = torch.sin(joint_pos[:, :, 1])
        s3 = torch.sin(joint_pos[:, :, 2])
        c1 = torch.cos(joint_pos[:, :, 0])
        c2 = torch.cos(joint_pos[:, :, 1])
        c3 = torch.cos(joint_pos[:, :, 2])
        c23 = c2 * c3 - s2 * s3
        s23 = s2 * c3 + c2 * s3

        if desired_joint_pos is None:
            self._foot_jacobian_b[:, :, 0, 0] = 0.0
            self._foot_jacobian_b[:, :, 1, 0] = - dy * s1 - dz2 * c1 * c23 - dz1 * c1 * c2
            self._foot_jacobian_b[:, :, 2, 0] = - dz2 * s1 * c23 + dy * c1 - dz1 * c2 * s1
            self._foot_jacobian_b[:, :, 0, 1] = dz2 * c23 + dz1 * c2
            self._foot_jacobian_b[:, :, 1, 1] = dz2 * s1 * s23 + dz1 * s1 * s2
            self._foot_jacobian_b[:, :, 2, 1] = - dz2 * c1 * s23 - dz1 * c1 * s2
            self._foot_jacobian_b[:, :, 0, 2] = dz2 * c23
            self._foot_jacobian_b[:, :, 1, 2] = dz2 * s1 * s23
            self._foot_jacobian_b[:, :, 2, 2] = - dz2 * c1 * s23
        else:
            desired_foot_jacobian_b = torch.zeros_like(self._foot_jacobian_b)
            desired_foot_jacobian_b[:, :, 0, 0] = 0.0
            desired_foot_jacobian_b[:, :, 1, 0] = - dy * s1 - dz2 * c1 * c23 - dz1 * c1 * c2
            desired_foot_jacobian_b[:, :, 2, 0] = - dz2 * s1 * c23 + dy * c1 - dz1 * c2 * s1
            desired_foot_jacobian_b[:, :, 0, 1] = dz2 * c23 + dz1 * c2
            desired_foot_jacobian_b[:, :, 1, 1] = dz2 * s1 * s23 + dz1 * s1 * s2
            desired_foot_jacobian_b[:, :, 2, 1] = - dz2 * c1 * s23 - dz1 * c1 * s2
            desired_foot_jacobian_b[:, :, 0, 2] = dz2 * c23
            desired_foot_jacobian_b[:, :, 1, 2] = dz2 * s1 * s23
            desired_foot_jacobian_b[:, :, 2, 2] = - dz2 * c1 * s23
            return desired_foot_jacobian_b

    def _compute_foot_pos_from_joint_pos(self, desired_joint_pos=None, return_frame='body'):
        joint_pos = self._joint_pos.clone() if desired_joint_pos is None else desired_joint_pos.clone()
        joint_pos = torch.cat((joint_pos[:, 0:3], joint_pos[:, 6:9], joint_pos[:, 12:]), dim=1) if self._use_gripper else joint_pos
        joint_pos = joint_pos.view(-1, 4, 3)

        theta_ab, theta_hip, theta_knee = joint_pos[..., 0], joint_pos[..., 1], joint_pos[..., 2]
        leg_distance = torch.sqrt(self._l_thi**2 + self._l_cal**2 + 2 * self._l_thi * self._l_cal * torch.cos(theta_knee))
        eff_swing = theta_hip + theta_knee / 2

        off_x_hip = -leg_distance * torch.sin(eff_swing)
        off_z_hip = -leg_distance * torch.cos(eff_swing)
        off_y_hip = self._l_hip * self.l_hip_sign * torch.ones_like(off_x_hip)

        off_x = off_x_hip
        off_y = torch.cos(theta_ab) * off_y_hip - torch.sin(theta_ab) * off_z_hip
        off_z = torch.sin(theta_ab) * off_y_hip + torch.cos(theta_ab) * off_z_hip

        foot_pos_hip = torch.stack([off_x, off_y, off_z], dim=-1)
        if return_frame == 'hip':
            return foot_pos_hip
        elif return_frame == 'body':
            foot_pos_b = foot_pos_hip + self._HIP_OFFSETS
            return foot_pos_b

    def _compute_foot_vel_from_joint_vel(self, desired_joint_vel=None, return_frame='body'):
        joint_vel = self._joint_vel.clone() if desired_joint_vel is None else desired_joint_vel.clone()
        joint_vel = torch.cat((joint_vel[:, 0:3], joint_vel[:, 6:9], joint_vel[:, 12:]), dim=1) if self._use_gripper else joint_vel
        joint_vel = joint_vel.view(-1, 4, 3, 1)

        foot_vel_local = torch.matmul(self._foot_jacobian_b, joint_vel).squeeze(-1)
        if return_frame == 'hip' or return_frame == 'body':
            return foot_vel_local
        else:
            print('return_frame must be hip or body!')

    def _forward_kinematics(self, joint_pos=None, joint_vel=None, return_frame='body'):
        foot_pos = self._compute_foot_pos_from_joint_pos(joint_pos, return_frame)
        foot_vel = self._compute_foot_vel_from_joint_vel(joint_vel, return_frame)
        return foot_pos, foot_vel

    # -------------------------access to robot state-------------------------
    @property
    def time_since_reset(self):
        return self._num_step * self._dt
    @property
    def time_since_reset_scalar(self):
        return self._num_step[0].cpu() * self._dt
    
    @property
    def torso_pos_w(self):
        return self._torso_pos_w
    @property
    def torso_quat_w2b(self):
        return self._torso_quat_w2b
    @property
    def torso_lin_vel_b(self):
        return self._torso_lin_vel_b
    @property
    def torso_lin_vel_w(self):
        return self._torso_lin_vel_w
    @property
    def torso_ang_vel_b(self):
        return self._torso_ang_vel_b
    @property
    def torso_ang_vel_w(self):
        return self._torso_ang_vel_w
    @property
    def torso_rot_mat_w2b(self):
        return self._torso_rot_mat_w2b
    @property
    def torso_rot_mat_b2w(self):
        return self._torso_rot_mat_b2w

    @property
    def torso_pos_w_np(self):
        return self._torso_pos_w.cpu().numpy()
    @property
    def torso_quat_w2b_np(self):
        return self._torso_quat_w2b.cpu().numpy()
    @property
    def torso_lin_vel_b_np(self):
        return self._torso_lin_vel_b.cpu().numpy()
    @property
    def torso_lin_vel_w_np(self):
        return self._torso_lin_vel_w.cpu().numpy()
    @property
    def torso_rpy_w2b_np(self):
        return self._torso_rpy_w2b.cpu().numpy()
    @property
    def torso_ang_vel_b_np(self):
        return self._torso_ang_vel_b.cpu().numpy()
    @property
    def torso_ang_vel_w_np(self):
        return self._torso_ang_vel_w.cpu().numpy()
    @property
    def torso_rot_mat_w2b_np(self):
        return self._torso_rot_mat_w2b.cpu().numpy()
    @property
    def torso_rot_mat_b2w_np(self):
        return self._torso_rot_mat_b2w.cpu().numpy()
    
    @property
    def joint_pos(self):
        return self._joint_pos
    @property
    def joint_vel(self):
        return self._joint_vel
    @property
    def joint_pos_np(self):
        return self._joint_pos.cpu().numpy()
    @property
    def joint_vel_np(self):
        return self._joint_vel.cpu().numpy()

    @property
    def foot_pos_hip(self):
        return self._foot_pos_hip
    @property
    def foot_vel_hip(self):
        return self._foot_vel_hip
    @property
    def foot_pos_b(self):
        return self._foot_pos_b
    @property
    def foot_vel_b(self):
        return self._foot_vel_b
    @property
    def foot_pos_w(self):
        return self._foot_pos_w
    @property
    def foot_vel_w(self):
        return self._foot_vel_w
    @property
    def origin_foot_pos_b(self):
        return self._origin_foot_pos_b
    @property
    def origin_foot_pos_hip(self):
        return self._origin_foot_pos_b - self._HIP_OFFSETS
    
    @property
    def foot_pos_hip_np(self):
        return self._foot_pos_hip.cpu().numpy()
    @property
    def foot_vel_hip_np(self):
        return self._foot_vel_hip.cpu().numpy()
    @property
    def foot_pos_b_np(self):
        return self._foot_pos_b.cpu().numpy()
    @property
    def foot_vel_b_np(self):
        return self._foot_vel_b.cpu().numpy()
    @property
    def foot_pos_w_np(self):
        return self._foot_pos_w.cpu().numpy()
    @property
    def foot_vel_w_np(self):
        return self._foot_vel_w.cpu().numpy()
    @property
    def origin_foot_pos_b_np(self):
        return self._origin_foot_pos_b.cpu().numpy()
    @property
    def origin_foot_pos_hip_np(self):
        return self.origin_foot_pos_hip.cpu().numpy()

    @property
    def foot_contact(self):
        return self._foot_contact
    @property
    def desired_foot_contact(self):
        return self._desired_foot_contact
    @property
    def foot_contact_np(self):
        return self._foot_contact.cpu().numpy()
    @property
    def desired_foot_contact_np(self):
        return self._desired_foot_contact.cpu().numpy()

    @property
    def foot_jacobian_b(self):
        return self._foot_jacobian_b
    @property
    def foot_jacobian_b_np(self):
        return self._foot_jacobian_b.cpu().numpy()
    
    @property
    def eef_pos_hip(self):
        return self._eef_pos_hip
    @property
    def eef_vel_hip(self):
        return self._eef_vel_hip
    @property
    def eef_pos_b(self):
        return self._eef_pos_b
    @property
    def eef_vel_b(self):
        return self._eef_vel_b
    @property
    def eef_pos_w(self):
        return self._eef_pos_w
    @property
    def eef_vel_w(self):
        return self._eef_vel_w
    @property
    def eef_jacobian_b(self):
        return self._eef_jacobian_b
    @property
    def eef_jacobian_w(self):
        return self._eef_jacobian_w
    @property
    def eef_rpy_w(self):
        return self._eef_rpy_w
    @property
    def eef_rot_w(self):
        return self._eef_rot_w
    @property
    def gripper_angles(self):
        return self._gripper_angles

    @property
    def eef_pos_hip_np(self):
        return self._eef_pos_hip.cpu().numpy()
    @property
    def eef_vel_hip_np(self):
        return self._eef_vel_hip.cpu().numpy()
    @property
    def eef_pos_b_np(self):
        return self._eef_pos_b.cpu().numpy()
    @property
    def eef_vel_b_np(self):
        return self._eef_vel_b.cpu().numpy()
    @property
    def eef_pos_w_np(self):
        return self._eef_pos_w.cpu().numpy()
    @property
    def eef_vel_w_np(self):
        return self._eef_vel_w.cpu().numpy()
    @property
    def eef_jacobian_b_np(self):
        return self._eef_jacobian_b.cpu().numpy()
    @property
    def eef_jacobian_w_np(self):
        return self._eef_jacobian_w.cpu().numpy()
    @property
    def eef_rpy_w_np(self):
        return self._eef_rpy_w.cpu().numpy()
    @property
    def eef_rot_w_np(self):
        return self._eef_rot_w.cpu().numpy()
    @property
    def gripper_angles_np(self):
        return self._gripper_angles.cpu().numpy()







