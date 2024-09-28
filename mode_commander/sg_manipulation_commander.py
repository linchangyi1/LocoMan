from mode_commander.base_commander import BaseCommander
import numpy as np
from fsm.finite_state_machine import FSM_State
from utilities.orientation_utils_numpy import rpy_to_rot_mat, rot_mat_to_rpy
import time
from scipy.spatial.transform import Rotation as R


class SingleGripperManipCommander(BaseCommander):
    def __init__(self, robot, env_ids=0):
        super().__init__(robot, env_ids=env_ids)
        self._wbc_command.operation_mode = FSM_State.SG_MANIPULATION
        self._manipulate_leg_idx = 0

    def reset(self):
        super().reset()
        self._manipulate_leg_idx = self._robot._cur_single_leg_idx.value
        self._wbc_command.contact_state[self._manipulate_leg_idx] = False
        self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 0:3] = self._robot.eef_pos_w_np[self._env_ids, self._manipulate_leg_idx]
        self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 3:6] = self._robot.eef_rpy_w_np[self._env_ids, self._manipulate_leg_idx]
        
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
            self._wbc_command_buffer.des_torso_pva[0, :] = command_np[0:6]
            self._wbc_command_buffer.des_gripper_pva[0, self._manipulate_leg_idx, :] = command_np[6:12]
            self._wbc_command_buffer.des_gripper_angles[self._manipulate_leg_idx] = command_np[12+self._manipulate_leg_idx]
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

                self._init_torso_xyz = self._wbc_command.des_torso_pva[0,0:3].copy()
                self._init_torso_rpy = self._wbc_command.des_torso_pva[0,3:6].copy()
                self._init_eef_xyz = self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 0:3].copy()
                self._init_eef_rpy = self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 3:6].copy()
                self._init_gripper_angle = self._wbc_command.des_gripper_angles[self._manipulate_leg_idx]

                self._last_torso_xyz_cmd = self._wbc_command.des_torso_pva[0,0:3].copy()
                self._last_torso_rpy_cmd = self._wbc_command.des_torso_pva[0,3:6].copy()
                self._last_eef_xyz_cmd = self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 0:3].copy()
                self._last_eef_rpy_cmd = self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 3:6].copy()
                self._last_gripper_angle_cmd = self._wbc_command.des_gripper_angles[self._manipulate_leg_idx]

                self._target_torso_xyz = self._wbc_command.des_torso_pva[0,0:3].copy()
                self._target_torso_rpy = self._wbc_command.des_torso_pva[0,3:6].copy()
                self._target_eef_xyz = self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 0:3].copy()
                self._target_eef_rotation_axis = np.ones_like(self._init_eef_rpy)
                self._target_eef_ratation_angle = 0.
                self._target_gripper_angle = self._wbc_command.des_gripper_angles[self._manipulate_leg_idx]

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
                self._last_torso_xyz_cmd[:] = self._wbc_command.des_torso_pva[0,0:3].copy()
                self._last_torso_rpy_cmd[:] = self._wbc_command.des_torso_pva[0,3:6].copy()
                self._last_eef_xyz_cmd[:] = self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 0:3].copy()
                self._last_eef_rpy_cmd[:] = self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 3:6].copy()
                self._last_gripper_angle_cmd = self._wbc_command.des_gripper_angles[self._manipulate_leg_idx]

                command_np[0:4] = 0.
                self._target_torso_xyz[:] = command_np[0:3] * self.torso_xyz_scale + self._init_torso_xyz
                self._target_torso_rpy[:] = command_np[3:6] * self.torso_rpy_scale + self._init_torso_rpy  
                self._target_eef_xyz[:] = command_np[6+6*self._manipulate_leg_idx:9+6*self._manipulate_leg_idx] * self.eef_xyz_scale + self._init_eef_xyz
                delta_rotation = rpy_to_rot_mat(command_np[9+6*self._manipulate_leg_idx:12+6*self._manipulate_leg_idx]) @ rpy_to_rot_mat(self._init_eef_rpy) @ (rpy_to_rot_mat(self._last_eef_rpy_cmd).T)
                r = R.from_matrix(delta_rotation)
                rotvec = r.as_rotvec()
                if np.linalg.norm(rotvec) > 1e-6:
                    self._target_eef_rotation_axis[:] = rotvec / np.linalg.norm(rotvec)
                    self._target_eef_ratation_angle = np.linalg.norm(rotvec) * self.eef_rpy_scale
                else:
                    self._target_eef_rotation_axis[:] = np.ones(3)
                    self._target_eef_ratation_angle = 0.
                self._target_gripper_angle = command_np[18+self._manipulate_leg_idx] * self.gripper_angle_scale + self._init_gripper_angle
            self._updating_buffer = False


    def compute_command_for_wbc(self):
        super().compute_command_for_wbc()
        self._accessing_buffer = True
        if self._time_for_tracking_human == 0.0:
            self._wbc_command.des_torso_pva[0, :] += self._wbc_command_buffer.des_torso_pva[0, :] * self._wbc_command_scale.des_torso_pva[0, :]
            self._wbc_command.des_torso_pva[0, :] = np.clip(self._wbc_command.des_torso_pva[0, :], -self._wbc_command_range.des_torso_pva[0, :], self._wbc_command_range.des_torso_pva[0, :])
            self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 0:3] += self._wbc_command_buffer.des_gripper_pva[0, self._manipulate_leg_idx, 0:3] * self._wbc_command_scale.des_gripper_pva[0, self._manipulate_leg_idx, 0:3]
            self._wbc_command_buffer.des_gripper_pva[0, self._manipulate_leg_idx, 3:6] *= self._wbc_command_scale.des_gripper_pva[0, self._manipulate_leg_idx, 3:6]
            if self._gripper_task_space_world:
                self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 3:6] = rot_mat_to_rpy(rpy_to_rot_mat(self._wbc_command_buffer.des_gripper_pva[0, self._manipulate_leg_idx, 3:6]) @ rpy_to_rot_mat(self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 3:6]))
            else:
                self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 3:6] = rot_mat_to_rpy(rpy_to_rot_mat(self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 3:6]) @ rpy_to_rot_mat(self._wbc_command_buffer.des_gripper_pva[0, self._manipulate_leg_idx, 3:6]))
            self._wbc_command.des_gripper_angles[self._manipulate_leg_idx] += self._wbc_command_buffer.des_gripper_angles[self._manipulate_leg_idx] * self._wbc_command_scale.des_gripper_angles[self._manipulate_leg_idx]
            self._wbc_command.des_gripper_angles[:] = np.clip(self._wbc_command.des_gripper_angles, self._wbc_command_range.des_gripper_angles[0], self._wbc_command_range.des_gripper_angles[1])
            self._wbc_command_buffer.reset(keep_mode=True)
        elif not self._first_receive_from_human:
            tracking_ratio = max(0.0, min(1.0, (time.time() - self._controller_time_at_first_stamp - self._controller_time_from_first_stamp) / self._time_for_tracking_human))

            self._wbc_command.des_torso_pva[0,0:3] += np.clip(self._last_torso_xyz_cmd * (1 - tracking_ratio) + self._target_torso_xyz * tracking_ratio - self._wbc_command.des_torso_pva[0,0:3], -self.torso_xyz_max_step, self.torso_xyz_max_step)
            self._wbc_command.des_torso_pva[0,3:6] += np.clip(self._last_torso_rpy_cmd * (1 - tracking_ratio) + self._target_torso_rpy * tracking_ratio - self._wbc_command.des_torso_pva[0,3:6], -self.torso_rpy_max_step, self.torso_rpy_max_step)
            self._wbc_command.des_torso_pva[0,0:5] = 0.
            self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 0:3] += np.clip(self._last_eef_xyz_cmd * (1 - tracking_ratio) + self._target_eef_xyz * tracking_ratio - self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 0:3], -self.eef_xyz_max_step, self.eef_xyz_max_step)

            rot_interpolated = R.from_rotvec(self._target_eef_ratation_angle*tracking_ratio*self._target_eef_rotation_axis).as_matrix()
            self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 3:6] = rot_mat_to_rpy(rot_interpolated @ rpy_to_rot_mat(self._last_eef_rpy_cmd))
            self._wbc_command.des_gripper_angles[self._manipulate_leg_idx] += max(-self.gripper_angle_max_step, min(self._last_gripper_angle_cmd * (1 - tracking_ratio) + self._target_gripper_angle * tracking_ratio - self._wbc_command.des_gripper_angles[self._manipulate_leg_idx], self.gripper_angle_max_step))
        self._accessing_buffer = False

        return self._wbc_command

