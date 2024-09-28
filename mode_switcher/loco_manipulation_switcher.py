import numpy as np
from robot.base_robot import BaseRobot
from scipy.spatial.transform import Rotation as R
from utilities.orientation_utils_numpy import rpy_to_rot_mat, rot_mat_to_rpy
from wbc.wbc_command import WBCCommand


class LocoManipulationSwitcher:
    def __init__(self, robot: BaseRobot, env_ids=0):
        self._robot = robot
        self._env_ids = env_ids
        self._cfg = robot._cfg

        self._manipulate_leg_idx = self._cfg.loco_manipulation.manipulate_leg_idx
        self._no_manipulate_leg_idx = 0 if self._manipulate_leg_idx == 1 else 1
        self._desired_eef_rpy_w = self._cfg.loco_manipulation.desired_eef_rpy_w.copy()
        self._transition_frames = int(self._cfg.switcher.stance_and_locomanipulation.transition_time / self._robot._dt)
        self._stablization_frames = int(self._cfg.switcher.stance_and_locomanipulation.stablization_time / self._robot._dt)

        self._wbc_command = WBCCommand()
        self._wbc_command.des_gripper_pva[0, 0, 3:6] = self._cfg.manipulator.reset_pos_sim[0:3]
        self._wbc_command.des_gripper_pva[0, 1, 3:6] = self._cfg.manipulator.reset_pos_sim[4:7]
        self._wbc_command.des_gripper_angles[0] = self._cfg.manipulator.reset_pos_sim[3]
        self._wbc_command.des_gripper_angles[1] = self._cfg.manipulator.reset_pos_sim[7]
        
        self._current_frame = 0
        self._current_eef_rot_w = rpy_to_rot_mat(self._robot.eef_rpy_w_np[self._env_ids, self._manipulate_leg_idx])
        delta_rotation = rpy_to_rot_mat(self._desired_eef_rpy_w) @ (self._current_eef_rot_w.T)
        r = R.from_matrix(delta_rotation)
        rotvec = r.as_rotvec()
        if np.linalg.norm(rotvec) > 1e-6:
            self._delta_eef_rotation_axis = rotvec / np.linalg.norm(rotvec)
            self._delta_eef_ratation_angle = np.linalg.norm(rotvec)
        else:
            self._delta_eef_rotation_axis = np.ones(3)
            self._delta_eef_ratation_angle = 0.

    def reset(self):
        self._current_frame = 0
        self._current_eef_rot_w[:] = rpy_to_rot_mat(self._robot.eef_rpy_w_np[self._env_ids, self._manipulate_leg_idx])
        delta_rotation = rpy_to_rot_mat(self._desired_eef_rpy_w) @ (self._current_eef_rot_w.T)
        r = R.from_matrix(delta_rotation)
        rotvec = r.as_rotvec()
        if np.linalg.norm(rotvec) > 1e-6:
            self._delta_eef_rotation_axis[:] = rotvec / np.linalg.norm(rotvec)
            self._delta_eef_ratation_angle = np.linalg.norm(rotvec)
        else:
            self._delta_eef_rotation_axis[:] = np.ones(3)
            self._delta_eef_ratation_angle = 0.

    def compute_command_for_wbc(self):
        self._current_frame += 1
        tracking_ratio = max(0.0, min(1.0, self._current_frame / self._transition_frames))
        rot_interpolated = R.from_rotvec(self._delta_eef_ratation_angle*tracking_ratio*self._delta_eef_rotation_axis).as_matrix()
        self._wbc_command.des_gripper_pva[0, self._manipulate_leg_idx, 3:6] = rot_mat_to_rpy(rot_interpolated @ self._current_eef_rot_w)
        return self._wbc_command


    def activate_switcher(self, stance_to_locomanipulation: bool):
        self._stance_to_locomanipulation = stance_to_locomanipulation
        if not self._stance_to_locomanipulation:
            self._loco_manipulation_commander = self._runner._fsm_commander._loco_manipulation_commander
            self._loco_manipulation_commander.prepare_to_stand()
        return self

    def feedback_execution(self, command_executed):
        pass

    def check_finished(self):
        return self._current_frame >= self._transition_frames + self._stablization_frames


