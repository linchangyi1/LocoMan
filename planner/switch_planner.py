import torch
import numpy as np
from robot.base_robot import BaseRobot
from planner.torso_foot_gripper_planner import TorsoFootGripperPlanner
from fsm.finite_state_machine import FSM_State
from wbc.wbc_command import WBCCommand


'''
The switch planner supports transitions between stance and either single-foot or single-gripper manipulation.
An intuitive approach to achieve these transitions is by hard-coding the trajectories of the base, foot, and manipulator.
However, this planner also allows the robot to perform sequences of actions simply by defining an array of actions,
such as tracking multiple 6D poses with interpolated trajectories.
'''
class SwitchPlanner:
    def __init__(self, robot: BaseRobot, action_sequences, manipulate_leg_idx, input_footgripper_frame='world', output_footgripper_frame='world', env_ids=0):
        self._robot = robot
        self._dt = self._robot._dt
        self._num_envs = self._robot._num_envs
        self._device = self._robot._device
        self._cfg = self._robot._cfg
        self._use_gripper = self._robot._use_gripper
        self._env_ids = env_ids
        self._input_footgripper_frame = input_footgripper_frame
        self._output_footgripper_frame = output_footgripper_frame
        self._manipulate_leg_idx = manipulate_leg_idx
        self._no_manipulate_leg_idx = 0 if self._manipulate_leg_idx==1 else 1
        self._torso_foot_gripper_planner = TorsoFootGripperPlanner(self._dt, self._num_envs, self._device, self._cfg.manipulation.torso_action_dim, self._cfg.manipulation.footgripper_pst_dim, self._cfg.manipulation.gripper_ori_dim)

        # initialize buffers
        self._all_env_idx = torch.arange(self._num_envs, device=self._device)
        self._current_action_idx = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._excute_or_during = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)  # excute: True, during: False
        self._finished_cycle_idx = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._contact_state = torch.ones((self._num_envs, 4), dtype=torch.bool, device=self._device)
        self._init_torso_state = torch.zeros((self._num_envs, self._cfg.manipulation.torso_action_dim), dtype=torch.float, device=self._device)
        self._origin_footgripper_psts = torch.zeros((self._num_envs, self._robot._num_legs, self._cfg.manipulation.footgripper_pst_dim), dtype=torch.float, device=self._device)
        self._init_footgripper_pst = torch.zeros((self._num_envs, self._cfg.manipulation.footgripper_pst_dim), dtype=torch.float, device=self._device)
        self._init_gripper_orts = torch.zeros((self._num_envs, len(self._cfg.asset.eef_names), self._cfg.manipulation.gripper_ori_dim), dtype=torch.float, device=self._device)
        if self._robot._cfg.commander.reset_manipulator_when_switch:
            self._gripper_angles_cmd = torch.ones((self._num_envs, len(self._robot._cfg.asset.eef_names)), dtype=torch.float, device=self._device) * self._cfg.manipulator.reset_pos_sim[3]
        else:
            self._gripper_angles_cmd = self._robot.gripper_angles.clone()
        self._last_desired_gripper_angles = self._gripper_angles_cmd.clone()
        self._next_desired_gripper_angles = self._gripper_angles_cmd.clone()

        # process the action sequences
        self._action_sequences = torch.tensor(action_sequences, dtype=torch.float, device=self._device)
        if self._output_footgripper_frame != self._input_footgripper_frame:
            self.manipulate_leg_state_transform()
        self._actions_num = self._action_sequences.shape[0]
        self._time_sequences = self._action_sequences[:, -2:].clone()
        self._gripper_angles = self._action_sequences[:, -4:-2].clone()
        self._reset_signal = self._action_sequences[:, -7].clone().to(torch.long)
        self._operation_mode = self._action_sequences[:, -8].clone().to(torch.long)
        # 0: manipulation to stance / no switch, 2: stance/sg-manipulation to sf-manipulation, 3: stance/sf-manipulation to sg-manipulation
        self._switch_mode = torch.zeros_like(self._operation_mode).to(torch.long)
        for i in range(self._actions_num):
            if i==0 and self._operation_mode[i]:  # the first action is always the stance
                self._switch_mode[i] = self._operation_mode[i]
            else:
                j = np.clip(i+1, 0, self._actions_num-1)
                if self._operation_mode[i] != self._operation_mode[j]:  # switch the mode if the next one is different
                    self._switch_mode[i] = self._operation_mode[j]

        # expand 3d footgripper pst to 12d foot psts for compatibility with the wbc_command
        self._wbc_foot_pva = torch.zeros((self._num_envs, 3, self._robot._num_legs*self._cfg.manipulation.footgripper_pst_dim), dtype=torch.float, device=self._device)
        self._manipulate_dofs = torch.tensor([3*self._manipulate_leg_idx, 3*self._manipulate_leg_idx+1, 3*self._manipulate_leg_idx+2], dtype=torch.long, device=self._device)
        self._wbc_command = WBCCommand()

    def manipulate_leg_state_transform(self):
        if self._input_footgripper_frame != 'world':
            locomotion_env_ids = (self._action_sequences[:, -8]==1.0).nonzero(as_tuple=False).flatten()
            if self._input_footgripper_frame == 'hip':
                leg_state_in_torso_frame = self._action_sequences[locomotion_env_ids, 6:9] + self._robot._HIP_OFFSETS[self._manipulate_leg_idx, :]
            else:
                leg_state_in_torso_frame = self._action_sequences[locomotion_env_ids, 6:9]
            if self._output_footgripper_frame == 'base':
                self._action_sequences[locomotion_env_ids, 6:9] = leg_state_in_torso_frame
            elif self._output_footgripper_frame == 'world':
                from utilities.rotation_utils import rpy_to_rot_mat
                T_w_b = torch.eye(4, dtype=torch.float, device=self._device).repeat(locomotion_env_ids.shape[0], 1, 1)
                T_w_b[:, :3, 3] = self._action_sequences[locomotion_env_ids, 0:3]
                T_w_b[:, :3, :3] = rpy_to_rot_mat(self._action_sequences[locomotion_env_ids, 3:6])
                self._action_sequences[locomotion_env_ids, 6:9] = torch.matmul(T_w_b, torch.cat([leg_state_in_torso_frame, torch.ones_like(leg_state_in_torso_frame[:, :1])], dim=-1).unsqueeze(-1)).squeeze(-1)[:, :3]

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = self._all_env_idx.clone()
        self._current_action_idx[env_ids] = 0
        self._excute_or_during[env_ids] = True
        self._finished_cycle_idx[env_ids] = False

        self._contact_state[env_ids, :] = True
        move_manipulate_foot_env_ids = (env_ids[(self._action_sequences[self._current_action_idx, -6][env_ids]==0.0) & (self._action_sequences[self._current_action_idx, -5][env_ids]==-1)])
        self._contact_state[move_manipulate_foot_env_ids, self._manipulate_leg_idx] = False
        move_non_manipulate_foot_env_ids = (env_ids[(self._action_sequences[self._current_action_idx, -6][env_ids]==0.0) & (self._action_sequences[self._current_action_idx, -5][env_ids]!=-1)])
        self._contact_state[move_non_manipulate_foot_env_ids, (self._action_sequences[self._current_action_idx[move_non_manipulate_foot_env_ids], -5]).to(torch.long)] = False

        self._save_origin_footgripper_psts(env_ids)
        self._update_initial_torso_state(env_ids)
        self._update_initial_footgripper_states(env_ids)

        self._torso_foot_gripper_planner.set_final_state(torso_final_state=self._action_sequences[self._current_action_idx, 0:6],
                                                        footgripper_final_pos=None,
                                                        gripper_final_oris=self._action_sequences[self._current_action_idx, 9:15],
                                                        action_duration=self._time_sequences[self._current_action_idx, 0],
                                                        env_ids=env_ids)

    def _save_origin_footgripper_psts(self, env_ids):
        if self._output_footgripper_frame == 'hip':
            self._origin_footgripper_psts[env_ids] = self._robot.origin_foot_pos_hip[env_ids, :, :]
        elif self._output_footgripper_frame == 'base':
            self._origin_footgripper_psts[env_ids] = self._robot.origin_foot_pos_b[env_ids, :, :]
        elif self._output_footgripper_frame == 'world':
            self._origin_footgripper_psts[env_ids] = self._robot._torso_pos_w[env_ids].unsqueeze(1) + torch.matmul(self._robot._torso_rot_mat_w2b[env_ids].unsqueeze(1), self._robot.origin_foot_pos_b[env_ids, :, :].unsqueeze(-1)).squeeze(-1)

    def _update_initial_torso_state(self, env_ids=None):
        if env_ids is None:
            env_ids = self._all_env_idx.clone()

        self._init_torso_state[env_ids] = torch.concatenate((self._robot._torso_pos_w[env_ids], self._robot._torso_rpy_w2b[env_ids]), dim=-1)
        self._torso_foot_gripper_planner.set_init_state(torso_init_state=self._init_torso_state,
                                                   footgripper_init_pos=None,
                                                   gripper_init_oris=None,
                                                   env_ids=env_ids)

    def _update_initial_footgripper_states(self, env_ids):
        if env_ids is None:
            env_ids = self._all_env_idx.clone()
        switch_mode = self._switch_mode[self._current_action_idx][env_ids]
        update_footgripper_state_env_ids = env_ids[(switch_mode==FSM_State.SF_MANIPULATION.value) | (switch_mode==FSM_State.SG_MANIPULATION.value)]
        if len(update_footgripper_state_env_ids)==0:
            return

        switch_to_foot_manipulation_env_ids = env_ids[(switch_mode==FSM_State.SF_MANIPULATION.value)]
        switch_to_gripper_manipulation_env_ids = env_ids[(switch_mode==FSM_State.SG_MANIPULATION.value)]

        if self._output_footgripper_frame == 'world':
            self._init_footgripper_pst[switch_to_foot_manipulation_env_ids] = self._robot.foot_pos_w[switch_to_foot_manipulation_env_ids, self._manipulate_leg_idx, :]
        elif self._output_footgripper_frame == 'base':
            self._init_footgripper_pst[switch_to_foot_manipulation_env_ids] = self._robot.foot_pos_b[switch_to_foot_manipulation_env_ids, self._manipulate_leg_idx, :]
        elif self._output_footgripper_frame == 'hip':
            self._init_footgripper_pst[switch_to_foot_manipulation_env_ids] = self._robot.foot_pos_hip[switch_to_foot_manipulation_env_ids, self._manipulate_leg_idx, :]

        if self._use_gripper:
            self._init_footgripper_pst[switch_to_gripper_manipulation_env_ids] = self._robot.eef_pos_w[switch_to_gripper_manipulation_env_ids, self._manipulate_leg_idx, :]
            self._init_gripper_orts[switch_to_foot_manipulation_env_ids, self._manipulate_leg_idx] = self._robot.joint_pos[switch_to_foot_manipulation_env_ids, 3+6*self._manipulate_leg_idx:6+6*self._manipulate_leg_idx]
            self._init_gripper_orts[switch_to_gripper_manipulation_env_ids, self._manipulate_leg_idx] = self._robot.eef_rpy_w[switch_to_gripper_manipulation_env_ids, self._manipulate_leg_idx, :]
            self._init_gripper_orts[env_ids, self._no_manipulate_leg_idx] = self._robot.joint_pos[env_ids, 3+6*self._no_manipulate_leg_idx:6+6*self._no_manipulate_leg_idx]

        self._torso_foot_gripper_planner.set_init_state(torso_init_state=None,
                                                   footgripper_init_pos=self._init_footgripper_pst,
                                                   gripper_init_oris=self._init_gripper_orts,
                                                   env_ids=update_footgripper_state_env_ids)

    def _set_final_states(self, env_ids):
        self._contact_state[env_ids, :] = True
        move_manipulate_foot_env_ids = (env_ids[(self._action_sequences[self._current_action_idx, -6][env_ids]==0.0) & (self._action_sequences[self._current_action_idx, -5][env_ids]==-1)])
        self._contact_state[move_manipulate_foot_env_ids, self._manipulate_leg_idx] = False
        move_non_manipulate_foot_env_ids = (env_ids[(self._action_sequences[self._current_action_idx, -6][env_ids]==0.0) & (self._action_sequences[self._current_action_idx, -5][env_ids]!=-1)])
        self._contact_state[move_non_manipulate_foot_env_ids, (self._action_sequences[self._current_action_idx[move_non_manipulate_foot_env_ids], -5]).to(torch.long)] = False

        torso_final_state = self._action_sequences[self._current_action_idx, 0:6].clone()
        torso_final_state[self._finished_cycle_idx] *= 0.0

        footgripper_final_pos = self._action_sequences[self._current_action_idx, 6:9].clone()
        footgripper_final_pos[move_non_manipulate_foot_env_ids, :] += self._origin_footgripper_psts[move_non_manipulate_foot_env_ids, self._action_sequences[self._current_action_idx, -5][move_non_manipulate_foot_env_ids].to(torch.long), :]

        full_contact_env_ids = env_ids[(self._action_sequences[self._current_action_idx[env_ids], -6]==1.0)]
        footgripper_final_pos[full_contact_env_ids, :] = self._torso_foot_gripper_planner._init_footgripper_pst[full_contact_env_ids, :]

        self._torso_foot_gripper_planner.set_final_state(torso_final_state=torso_final_state,
                                                    footgripper_final_pos=footgripper_final_pos,
                                                    gripper_final_oris=self._action_sequences[self._current_action_idx, 9:15],
                                                    action_duration=self._time_sequences[self._current_action_idx, (~self._excute_or_during).to(torch.long)],
                                                    env_ids=env_ids)

    def compute_command_for_wbc(self):
        env_action_end = self._torso_foot_gripper_planner.step()  # phase=1.0

        during_env_ids = (self._excute_or_during==False)
        if during_env_ids.any():
            current_phase = self._torso_foot_gripper_planner._current_phase[during_env_ids].unsqueeze(1)
            last_angles = self._last_desired_gripper_angles[during_env_ids, :]
            next_angles = self._next_desired_gripper_angles[during_env_ids, :]
            self._gripper_angles_cmd[during_env_ids, :] = ((1 - current_phase) * last_angles + current_phase * next_angles)

        env_action_end &= (~self._finished_cycle_idx)  # don't update the environments that have finished the cycle
        if torch.sum(env_action_end) != 0:
            update_gripper_action_env_ids = env_action_end & (self._excute_or_during==True)
            if torch.sum(update_gripper_action_env_ids) != 0:
                self._last_desired_gripper_angles[update_gripper_action_env_ids] = self._gripper_angles_cmd[update_gripper_action_env_ids]
                self._next_desired_gripper_angles[update_gripper_action_env_ids] = self._gripper_angles[self._current_action_idx[update_gripper_action_env_ids]]

            # must compute the enviroments that need to be reset before updating the general items
            reset_robot_env_ids = env_action_end & (self._action_sequences[self._current_action_idx, -7]==1.0) & (self._excute_or_during==False)
            if torch.sum(reset_robot_env_ids) != 0:
                reset_robot_env_ids = reset_robot_env_ids.nonzero(as_tuple=False).flatten()
                self._robot._update_state(reset_estimator=True, env_ids=reset_robot_env_ids)
                self._save_origin_footgripper_psts(env_ids=reset_robot_env_ids)
                self._update_initial_torso_state(env_ids=reset_robot_env_ids)
                self._update_initial_footgripper_states(env_ids=reset_robot_env_ids)

            update_initial_footgripper_state_env_ids = (env_action_end & (self._excute_or_during==False)).nonzero(as_tuple=False).flatten()

            # update the general items
            self._excute_or_during[env_action_end] = ~self._excute_or_during[env_action_end]
            self._finished_cycle_idx |= env_action_end & (self._current_action_idx==(self._actions_num-1)) & (self._excute_or_during==True)
            self._current_action_idx[env_action_end & self._excute_or_during] += 1
            self._current_action_idx = torch.clip(self._current_action_idx, 0, self._actions_num-1)

            # update the initial and final states of the planer
            env_action_end = env_action_end.nonzero(as_tuple=False).flatten()
            self._update_initial_footgripper_states(env_ids=update_initial_footgripper_state_env_ids)
            self._set_final_states(env_ids=env_action_end)

        self._wbc_command.operation_mode = FSM_State(self._operation_mode[self._current_action_idx][self._env_ids].cpu().numpy())
        self._wbc_command.contact_state[:] = self.get_contact_state()[self._env_ids].cpu().numpy()
        self._wbc_command.des_torso_pva[:] = self.get_desired_torso_pva()[self._env_ids].cpu().numpy().reshape(3, 6)
        self._wbc_command.des_foot_pva[:] = self.get_desired_foot_pva()[self._env_ids].cpu().numpy().reshape(3, 4, 3)
        if self._use_gripper:
            if self._wbc_command.operation_mode == FSM_State.SG_MANIPULATION:
                manipulate_idx = np.nonzero(np.logical_not(self._wbc_command.contact_state))[0]
                self._wbc_command.des_gripper_pva[:, manipulate_idx, 0:3] = self._wbc_command.des_foot_pva[:, manipulate_idx, :]
            self._wbc_command.des_gripper_pva[0, :, 3:6] = self.get_desired_gripper_ori_pva()[self._env_ids].cpu().numpy()[:, 0:3]
            self._wbc_command.des_gripper_pva[1, :, 3:6] = self.get_desired_gripper_ori_pva()[self._env_ids].cpu().numpy()[:, 3:6]
            self._wbc_command.des_gripper_pva[2, :, 3:6] = self.get_desired_gripper_ori_pva()[self._env_ids].cpu().numpy()[:, 6:9]
        self._wbc_command.des_gripper_angles[:] = self._gripper_angles_cmd[self._env_ids].cpu().numpy()

        return self._wbc_command


    def feedback_execution(self, command_executed):
        pass

    def check_finished(self):
        return torch.sum(self._finished_cycle_idx) > 0

    def get_desired_torso_pva(self):
        return self._torso_foot_gripper_planner.get_desired_torso_pva()
    
    def get_desired_foot_pva(self):
        self._wbc_foot_pva[:, :, self._manipulate_dofs] = self._torso_foot_gripper_planner.get_desired_footgripper_pva().reshape(self._num_envs, 3, -1)
        return self._wbc_foot_pva

    def get_desired_gripper_ori_pva(self):
        return self._torso_foot_gripper_planner.get_desired_gripper_ori_pva()

    def get_contact_state(self):
        return self._contact_state
    
    def get_gripper_angles_cmd(self):
        return self._gripper_angles_cmd


