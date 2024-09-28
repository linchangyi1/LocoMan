from mode_commander.base_commander import BaseCommander
import numpy as np
from utilities.orientation_utils_numpy import rpy_to_rot_mat, rot_mat_to_rpy
import time
from scipy.spatial.transform import Rotation as R
from fsm.finite_state_machine import FSM_State


class BiManipCommander(BaseCommander):
    def __init__(self, robot, env_ids=0):
        super().__init__(robot, env_ids=env_ids)
        self._wbc_command.operation_mode = FSM_State.BIMANIPULATION
        self._manipulate_leg_idx = [0, 1]
        self._wbc_command.rear_legs_pos = np.array([0, 2.35, -2.17, 0, 2.35, -2.17])
        self._wbc_command.rear_legs_vel = np.zeros(6)
        self._wbc_command.rear_legs_torque = np.zeros(6)

    def reset(self):
        super().reset()
        self._wbc_command.contact_state[self._manipulate_leg_idx] = False
        self._wbc_command.des_torso_pva[0, 3:6] = self._robot.torso_rpy_w2b_np[self._env_ids]
        if self._use_gripper:
            self._wbc_command.des_gripper_pva[0, :, 0:3] = self._robot.eef_pos_w_np[self._env_ids]
            self._wbc_command.des_gripper_pva[0, :, 3:6] = self._robot.eef_rpy_w_np[self._env_ids]
        else:
            self._wbc_command.des_foot_pva[0, self._manipulate_leg_idx, :] = self._robot.foot_pos_w_np[self._env_ids, self._manipulate_leg_idx]            

        self._human_time_at_first_stamp = time.time()
        self._human_time_from_first_stamp = 0.
        self._controller_time_at_first_stamp = time.time()
        self._controller_time_from_first_stamp = 0.
        self._reset_from_human = True

    def _update_joystick_command_callback(self, command_msg):
        if self._commander_active:
            while self._accessing_buffer:
                pass
            self._updating_buffer = True
            self._time_for_tracking_human = 0.0
            command_np = np.array(command_msg.data)
            if self._use_gripper:
                self._wbc_command_buffer.des_gripper_pva[0, :, :] = command_np[0:12].reshape(2, 6)
                self._wbc_command_buffer.des_gripper_angles[:] = command_np[12:14]
            else:
                self._wbc_command_buffer.des_foot_pva[0, 1, :] = command_np[6:9]
                self._wbc_command_buffer.des_foot_pva[0, 0, 0] = -command_np[10]
                self._wbc_command_buffer.des_foot_pva[0, 0, 1] = command_np[11]
                self._wbc_command_buffer.des_foot_pva[0, 0, 2] = command_np[9]
            self._updating_buffer = False
            for _ in range(10):
                pass

    def _update_human_command_callback(self, command_msg):
        if self._commander_active:
            self._updating_buffer = True
            if self._reset_from_human:
                self._human_time_at_first_stamp = time.time()
                self._human_time_from_first_stamp = 0.
                self._controller_time_at_first_stamp = time.time()
                self._controller_time_from_first_stamp = 0.

                self._init_eefs_xyz = self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 0:3].copy() if self._use_gripper else self._wbc_command.des_foot_pva[0, self._manipulate_leg_idx, :].copy()
                self._init_eefs_rpy = self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 3:6].copy() if self._use_gripper else np.zeros(3)
                self._init_gripper_angles = self._wbc_command.des_gripper_angles.copy()

                self._last_eefs_xyz_cmd = self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 0:3].copy() if self._use_gripper else self._wbc_command.des_foot_pva[0, self._manipulate_leg_idx, :].copy()
                self._last_eefs_rpy_cmd = self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 3:6].copy() if self._use_gripper else np.zeros(3)
                self._last_gripper_angles_cmd = self._wbc_command.des_gripper_angles.copy()

                self._target_eefs_xyz = self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 0:3].copy() if self._use_gripper else self._wbc_command.des_foot_pva[0, self._manipulate_leg_idx, :].copy()
                self._target_eefs_rotation_axis = np.ones_like(self._init_eefs_rpy)
                self._target_eefs_ratation_angle = np.zeros(2)
                self._target_gripper_angles = self._wbc_command.des_gripper_angles.copy()

                self._reset_from_human = False
            else:
                self._human_time_from_first_stamp = time.time() - self._human_time_at_first_stamp
                if self._first_receive_from_human:
                    self._controller_time_at_first_stamp = time.time()
                    self._first_receive_from_human = False
                else:
                    self._controller_time_from_first_stamp = time.time() - self._controller_time_at_first_stamp
                self._time_for_tracking_human = self._human_time_from_first_stamp - self._controller_time_from_first_stamp

                command_np = np.array(command_msg.data)
                self._last_eefs_xyz_cmd = self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 0:3].copy() if self._use_gripper else self._wbc_command.des_foot_pva[0, self._manipulate_leg_idx, :].copy()
                self._last_eefs_rpy_cmd = self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 3:6].copy() if self._use_gripper else np.zeros(3)
                self._last_gripper_angles_cmd = self._wbc_command.des_gripper_angles.copy()

                for i in self._manipulate_leg_idx:
                    self._target_eefs_xyz[i, :] = self._init_eefs_xyz[i] + command_np[6+6*i:9+6*i] * self.eef_xyz_scale
                    delta_rotation = rpy_to_rot_mat(command_np[9+6*i:12+6*i]) @ rpy_to_rot_mat(self._init_eefs_rpy[i, :]) @ (rpy_to_rot_mat(self._last_eefs_rpy_cmd[i, :]).T)

                    r = R.from_matrix(delta_rotation)
                    rotvec = r.as_rotvec()
                    if np.linalg.norm(rotvec) > 1e-6:
                        self._target_eefs_rotation_axis[i, :] = rotvec / np.linalg.norm(rotvec)
                        self._target_eefs_ratation_angle[i] = np.linalg.norm(rotvec) * self.eef_rpy_scale
                        if i==0:
                            self._target_eefs_ratation_angle[i] = 0.4 if self._target_eefs_ratation_angle[i] > 0.4 else self._target_eefs_ratation_angle[i]
                        else:
                            self._target_eefs_ratation_angle[i] = -0.4 if self._target_eefs_ratation_angle[i] < -0.4 else self._target_eefs_ratation_angle[i]
                    else:
                        self._target_eefs_rotation_axis[i, :] = np.ones(3)
                        self._target_eefs_ratation_angle[i] = 0.
                    command_np[18+i] = 0. if command_np[18+i] < 0.2 else command_np[18+i]-0.2
                    self._target_gripper_angles[i] = command_np[18+i] * self.gripper_angle_scale + self._init_gripper_angles[i]

            self._updating_buffer = False

    def compute_command_for_wbc(self):
        super().compute_command_for_wbc()
        self._accessing_buffer = True
        if self._time_for_tracking_human == 0.0:
            if self._use_gripper:
                self._wbc_command.des_gripper_pva[0, :, 0:3] += self._wbc_command_buffer.des_gripper_pva[0, :, 0:3] * self._wbc_command_scale.des_gripper_pva[0, :, 0:3]
                self._wbc_command_buffer.des_gripper_pva[0, :, 3:6] *= self._wbc_command_scale.des_gripper_pva[0, :, 3:6]
                self._wbc_command.des_gripper_angles[:] += self._wbc_command_buffer.des_gripper_angles * self._wbc_command_scale.des_gripper_angles
                self._wbc_command.des_gripper_angles[:] = np.clip(self._wbc_command.des_gripper_angles, self._wbc_command_range.des_gripper_angles[0], self._wbc_command_range.des_gripper_angles[1])
                for i in self._manipulate_leg_idx:
                    if self._gripper_task_space_world:
                        self._wbc_command.des_gripper_pva[0, i, 3:6] = rot_mat_to_rpy(rpy_to_rot_mat(self._wbc_command_buffer.des_gripper_pva[0, i, 3:6]) @ rpy_to_rot_mat(self._wbc_command.des_gripper_pva[0, i, 3:6]))
                    else:
                        self._wbc_command.des_gripper_pva[0, i, 3:6] = rot_mat_to_rpy(rpy_to_rot_mat(self._wbc_command.des_gripper_pva[0, i, 3:6]) @ rpy_to_rot_mat(self._wbc_command_buffer.des_gripper_pva[0, i, 3:6]))
            else:
                self._wbc_command.des_foot_pva[0, self._manipulate_leg_idx, :] += self._wbc_command_buffer.des_foot_pva[0, self._manipulate_leg_idx, :] * self._wbc_command_scale.des_foot_pva[0, self._manipulate_leg_idx, :]
            self._wbc_command_buffer.reset(keep_mode=True)
        elif not self._first_receive_from_human:
            tracking_ratio = max(1.0, (time.time() - self._controller_time_at_first_stamp - self._controller_time_from_first_stamp) / self._time_for_tracking_human)
            if self._use_gripper:
                self._wbc_command.des_gripper_pva[0, :, 0:3] += np.clip(self._last_eefs_xyz_cmd * (1 - tracking_ratio) + self._target_eefs_xyz * tracking_ratio - self._wbc_command.des_gripper_pva[0, :, 0:3], -self.eef_xyz_max_step, self.eef_xyz_max_step)
                self._wbc_command.des_gripper_angles[:] += max(-self.gripper_angle_max_step, min(self._last_gripper_angles_cmd * (1 - tracking_ratio) + self._target_gripper_angles * tracking_ratio - self._wbc_command.des_gripper_angles, self.gripper_angle_max_step))
                for i in self._manipulate_leg_idx:
                    rot_interpolated = R.from_rotvec(self._target_eefs_ratation_angle[i]*tracking_ratio*self._target_eefs_rotation_axis[i]).as_matrix()
                    self._wbc_command.des_gripper_pva[0, i, 3:6] = rot_mat_to_rpy(rot_interpolated @ rpy_to_rot_mat(self._last_eefs_rpy_cmd[i]))
            else:
                self._wbc_command.des_foot_pva[0, self._manipulate_leg_idx, :] += np.clip(self._last_eefs_xyz_cmd * (1 - tracking_ratio) + self._target_eefs_xyz * tracking_ratio - self._wbc_command.des_foot_pva[0, self._manipulate_leg_idx, :], -self.eef_xyz_max_step, self.eef_xyz_max_step)
        self._accessing_buffer = False

        return self._wbc_command
    
