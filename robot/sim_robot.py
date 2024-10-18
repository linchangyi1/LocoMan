from config.config import Cfg
from robot.base_robot import BaseRobot
from robot.motors import MotorCommand, MotorControlMode
from isaacgym import gymapi, gymtorch
from utilities.rotation_utils import rpy_vel_to_skew_synmetric_mat
import numpy as np
import torch
import os
import sys



class SimRobot(BaseRobot):
    def __init__(self, cfg: Cfg, sim, viewer):
        super().__init__(cfg)
        self._gym = gymapi.acquire_gym()
        self._sim = sim
        self._viewer = viewer
        self._sim_conf = self._cfg.get_sim_config()
        self._init_simulator()
        self._init_buffers()

    def _init_simulator(self):
        self._prepare_initial_state()
        self._create_envs()
        self._gym.prepare_sim(self._sim)
    
    def _prepare_initial_state(self):
        torso_init_state_list = self._cfg.init_state.pos + self._cfg.init_state.rot + self._cfg.init_state.lin_vel + self._cfg.init_state.ang_vel
        torso_init_states = np.stack([torso_init_state_list] * self._num_envs, axis=0)
        self._torso_init_state = torch.tensor(torso_init_states, dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_init_pos = self._motors.init_positions
        if "cuda" in self._device:
            torch._C._jit_set_profiling_mode(False)
            torch._C._jit_set_profiling_executor(False)

    def _create_envs(self):
        # Load robot asset
        urdf_path = self._cfg.asset.urdf_path
        asset_root = os.path.dirname(urdf_path)
        asset_file = os.path.basename(urdf_path)
        asset_config = self._cfg.get_asset_config()
        self._robot_asset = self._gym.load_asset(self._sim, asset_root, asset_file, asset_config.asset_options)

        # Create envs and actors
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self._envs = []
        self._actors = []
        for i in range(self._num_envs):
            env_handle = self._gym.create_env(self._sim, env_lower, env_upper, int(np.sqrt(self._num_envs)))
            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(*self._torso_init_state[i, :3])
            actor_handle = self._gym.create_actor(env_handle, self._robot_asset, start_pose, "actor",
                                                  i, asset_config.self_collisions, 0)
            self._gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
            self._envs.append(env_handle)
            self._actors.append(actor_handle)
    
    def _init_buffers(self):
        # Rigid body indices within all rigid bodies
        self._torso_indices = torch.zeros(len(self._torso_names), dtype=torch.long, device=self._device, requires_grad=False)
        self._hip_indices = torch.zeros(len(self._hip_names), dtype=torch.long, device=self._device, requires_grad=False)
        self._thigh_indices = torch.zeros(len(self._thigh_names), dtype=torch.long, device=self._device, requires_grad=False)
        self._calf_indices = torch.zeros(len(self._calf_names), dtype=torch.long, device=self._device, requires_grad=False)
        self._foot_indices = torch.zeros(self._num_legs, dtype=torch.long, device=self._device, requires_grad=False)

        # Extract indices of different bodies
        for i in range(len(self._torso_names)):
            self._torso_indices[i] = self._gym.find_actor_rigid_body_handle(self._envs[0], self._actors[0], self._torso_names[i])
        for i in range(len(self._hip_names)):
            self._hip_indices[i] = self._gym.find_actor_rigid_body_handle(self._envs[0], self._actors[0], self._hip_names[i])
        for i in range(len(self._thigh_names)):
            self._thigh_indices[i] = self._gym.find_actor_rigid_body_handle(self._envs[0], self._actors[0], self._thigh_names[i])
        for i in range(len(self._calf_names)):
            self._calf_indices[i] = self._gym.find_actor_rigid_body_handle(self._envs[0], self._actors[0], self._calf_names[i])
        for i in range(self._num_legs):
            self._foot_indices[i] = self._gym.find_actor_rigid_body_handle(self._envs[0], self._actors[0], self._foot_names[i])
        
        # Get gym GPU state tensors
        actor_root_state = self._gym.acquire_actor_root_state_tensor(self._sim)
        dof_state_tensor = self._gym.acquire_dof_state_tensor(self._sim)
        net_contact_forces = self._gym.acquire_net_contact_force_tensor(self._sim)
        rigid_body_state = self._gym.acquire_rigid_body_state_tensor(self._sim)
        jacobians = self._gym.acquire_jacobian_tensor(self._sim, "actor")
        massmatrix = self._gym.acquire_mass_matrix_tensor(self._sim, "actor")
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        self._gym.refresh_jacobian_tensors(self._sim)
        self._gym.refresh_mass_matrix_tensors(self._sim)

        # Robot state buffers
        # root state
        self._root_state = gymtorch.wrap_tensor(actor_root_state)
        self._torso_pos_sim = self._root_state[:, 0:3]
        self._torso_quat_sim2b = self._root_state[:, 3:7]
        self._torso_lin_vel_sim = self._root_state[:, 7:10]
        self._torso_ang_vel_sim = self._root_state[:, 10:13]

        # dof state
        self._num_joints = self._gym.get_asset_dof_count(self._robot_asset)
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._joint_pos = self._dof_state.view(self._num_envs, self._num_joints, 2)[..., 0]
        self._joint_vel = self._dof_state.view(self._num_envs, self._num_joints, 2)[..., 1]
        # force state
        self._contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self._num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self._foot_contact = self._contact_forces[:, self._foot_indices, 2] > 1.0
        self._jacobian_sim = gymtorch.wrap_tensor(jacobians)
        self._mass_matrix = gymtorch.wrap_tensor(massmatrix)
        self._gravity_vec = torch.stack([torch.tensor([0., 0., 1.], dtype=torch.float, device=self._device, requires_grad=False)] * self._num_envs)
        self._projected_gravity = torch.bmm(self._torso_rot_mat_b2w, self._gravity_vec[:, :, None])[:, :, 0]
        # rigid body state
        self._num_bodies = self._gym.get_asset_rigid_body_count(self._robot_asset)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self._num_envs * self._num_bodies, :].view(self._num_envs, self._num_bodies, 13)
        self._foot_pos_sim = self._rigid_body_state[:, self._foot_indices, 0:3]
        self._foot_vel_sim = self._rigid_body_state[:, self._foot_indices, 7:10]

        if self._use_gripper:
            self._eef_indices = torch.zeros(len(self._eef_names), dtype=torch.long, device=self._device, requires_grad=False)
            for i in range(len(self._eef_names)):
                self._eef_indices[i] = self._gym.find_actor_rigid_body_handle(self._envs[0], self._actors[0], self._eef_names[i])
            self._eef_pos_sim = self._rigid_body_state[:, self._eef_indices, 0:3]
            self._eef_quat_sim = self._rigid_body_state[:, self._eef_indices, 3:7]

        # torque
        self._torques = torch.zeros(self._num_envs, self._num_joints, dtype=torch.float, device=self._device, requires_grad=False)

        # The origins for each environment
        num_cols = np.floor(np.sqrt(self._num_envs))
        num_rows = np.ceil(self._num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing='ij')
        spacing = self._sim_conf.env_spacing
        self._env_origins = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._env_origins[:, 0] = spacing * xx.to(self._device).flatten()[0:self._num_envs]
        self._env_origins[:, 1] = spacing * yy.to(self._device).flatten()[0:self._num_envs]
        self._env_origins[:, 2] = 0.

        # useful buffers
        if self._cfg.sim.show_gui:
            self.first_render = True
        self.log_one_time = True
        self.finish_reset = False

    def reset(self):
        self.finish_reset = False
        self._reset_idx(torch.arange(self._num_envs, device=self._device))
        self.finish_reset = True
    
    def _reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # Reset root state:
        self._root_state[env_ids] = self._torso_init_state[env_ids]
        self._root_state[env_ids, :3] += self._env_origins[env_ids]
        self._gym.set_actor_root_state_tensor_indexed(self._sim, gymtorch.unwrap_tensor(self._root_state),
                                                      gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # Reset dofs
        self._joint_pos[env_ids] = self._joint_init_pos
        self._joint_vel[env_ids] = 0.
        self._gym.set_dof_state_tensor_indexed(self._sim, gymtorch.unwrap_tensor(self._dof_state),
                                               gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # If reset all enviroments
        if len(env_ids) == self._num_envs:
            self._stablize_the_robot()
            self._num_step[:] = 0
        else:
            self._num_step[env_ids] = 0
        self._update_state(reset_estimator=True, env_ids=env_ids)

    def _stablize_the_robot(self):
        print("Ready to reset the robot!")
        zero_action = MotorCommand(desired_position=self._joint_init_pos.repeat(self._num_envs, 1),
                                    kp=self._motors.kps,
                                    desired_velocity=torch.zeros_like(self._joint_init_pos),
                                    kd=self._motors.kds)
        for _ in torch.arange(0, self._cfg.motor_control.reset_time, self._dt):
            self.step(zero_action, MotorControlMode.POSITION)
        print("Robot reset done!")
    
    def step(self, action: MotorCommand, motor_control_mode: MotorControlMode = None, gripper_cmd=True):
        self._log_info_now = self._log_info and self._num_step[0] % self._log_interval == 0
        self._num_step[:] += 1
        for _ in range(self._sim_conf.action_repeat):
            self._apply_action(action, motor_control_mode)
            self._gym.refresh_dof_state_tensor(self._sim)  # only need to refresh dof state for updating reference joit position and velocity
        self._update_state()
        if self._cfg.sim.show_gui:
            self._render()

    def _apply_action(self, action: MotorCommand, motor_control_mode: MotorControlMode = None):
        self._action_to_torque(action, motor_control_mode)
        self._gym.set_dof_actuation_force_tensor(self._sim, gymtorch.unwrap_tensor(self._torques))
        self._gym.simulate(self._sim)
        if self._device == "cpu":
            self._gym.fetch_results(self._sim, True)

    def _action_to_torque(self, action: MotorCommand, motor_control_mode: MotorControlMode = None):
        if motor_control_mode is None:
            motor_control_mode = self._motors._motor_control_mode
        if motor_control_mode == MotorControlMode.POSITION:
            self._torques[:] = action.kp * (action.desired_position - self._joint_pos) - action.kd * self._joint_vel
        elif motor_control_mode == MotorControlMode.TORQUE:
            self._torques[:] = action.desired_extra_torque
        elif motor_control_mode == MotorControlMode.HYBRID:
            self._torques[:] = action.kp * (action.desired_position - self._joint_pos) +\
                               action.kd * (action.desired_velocity - self._joint_vel) +\
                               action.desired_extra_torque
        else:
            raise ValueError('Unknown motor control mode for Go1 robot: {}.'.format(motor_control_mode))

    def _update_sensors(self):
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        self._gym.refresh_dof_force_tensor(self._sim)
        self._gym.refresh_jacobian_tensors(self._sim)
        self._gym.refresh_mass_matrix_tensors(self._sim)
        self._foot_pos_sim = self._rigid_body_state[:, self._foot_indices, 0:3]
        self._foot_vel_sim = self._rigid_body_state[:, self._foot_indices, 7:10]
        if self._use_gripper:
            self._eef_pos_sim = self._rigid_body_state[:, self._eef_indices, 0:3]
            self._eef_quat_sim = self._rigid_body_state[:, self._eef_indices, 3:7]

    def _update_foot_global_state(self):
        self._state_estimator[self._cur_fsm_state].set_foot_global_state()

    def _update_foot_contact_state(self):
        self._foot_contact[:] = self._contact_forces[:, self._foot_indices, 2] > 1.0

    def _update_foot_jocabian_position_velocity(self):
        # compute the jacobian in the world frame
        self._state_estimator[self._cur_fsm_state].compute_jacobian_w()

        # compute the foot jacobian in the body frame
        for i in range(self._num_legs):
            self._foot_jacobian_b[:, i, :, :] = torch.bmm(self._torso_rot_mat_b2w, self._jacobian_w[:, self._foot_indices[i], :3, 6:9])

        # compute foot local position
        self._foot_pos_b[:] = torch.bmm(self._torso_rot_mat_b2w, (self._foot_pos_w-self._torso_pos_w.unsqueeze(1)).transpose(1, 2)).transpose(1, 2)
        self._foot_pos_hip[:] = self._foot_pos_b - self._HIP_OFFSETS

        # compute foot local velocity
        # Vf^b = R_b^w * (Vf^w - Vb^w - [w_b^w] * R_w^b * pf^b)
        self._foot_vel_b[:] = torch.bmm(self._torso_rot_mat_b2w,
                                        (self._foot_vel_w - self._torso_lin_vel_w.unsqueeze(1)).transpose(-2, -1) -\
                                            torch.bmm(rpy_vel_to_skew_synmetric_mat(self._torso_ang_vel_w), torch.bmm(self._torso_rot_mat_b2w.transpose(-2, -1), self._foot_pos_b.transpose(-2, -1)))).transpose(1, 2)
        self._foot_vel_hip[:] = self._foot_vel_b

    def _render(self, sync_frame_time=True):
        if self._viewer:
            if self._gym.query_viewer_has_closed(self._viewer):
                sys.exit()

        if self.first_render:
            direction_scale = 1.0
            mean_pos = torch.min(self._torso_pos_w,
                                dim=0)[0].cpu().numpy() + np.array([.8, direction_scale*-.9, .4])
            target_pos = torch.mean(self._torso_pos_w, dim=0).cpu().numpy() + np.array([0.2, 0., 0.])
            cam_pos = gymapi.Vec3(*mean_pos)
            cam_target = gymapi.Vec3(*target_pos)
            self._gym.viewer_camera_look_at(self._viewer, None, cam_pos, cam_target)
            self.first_render = False

        if self._device != "cpu":
            self._gym.fetch_results(self._sim, True)

        self._gym.step_graphics(self._sim)
        self._gym.draw_viewer(self._viewer, self._sim, True)
        if sync_frame_time:
            self._gym.sync_frame_time(self._sim)

    # torques
    @property
    def torques(self):
        return self._torques
