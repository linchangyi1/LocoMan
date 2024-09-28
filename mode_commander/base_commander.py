import numpy as np
from robot.base_robot import BaseRobot
from config.config import Cfg
import rospy
from std_msgs.msg import Float32MultiArray
import time
from wbc.wbc_command import WBCCommand


class BaseCommander:
    def __init__(self, robot: BaseRobot, env_ids=0):
        self._robot = robot
        self._cfg: Cfg = robot._cfg
        self._dt = self._robot._dt
        self._env_ids = env_ids
        self._use_gripper = self._robot._use_gripper
        self._commander_active = False
        self._updating_buffer = False
        self._accessing_buffer = False
        self._command_executed = True

        # new for wbc_command
        self._wbc_command = WBCCommand()
        self._wbc_command_scale = WBCCommand()
        self._wbc_command_range = WBCCommand()
        self._wbc_command_buffer = WBCCommand()
        self._wbc_command_last_executed = WBCCommand()

        self._wbc_command_scale.des_torso_pva[0:2, :] = self._cfg.commander.torso_pv_scale.reshape(2, 6)
        self._wbc_command_scale.des_foot_pva[0, :, :] = self._cfg.commander.foot_xyz_scale
        self._wbc_command_scale.des_gripper_pva[0, :, 0:3] = self._cfg.commander.gripper_xyz_scale
        self._wbc_command_scale.des_gripper_pva[0, :, 3:6] = self._cfg.commander.gripper_rpy_scale
        self._wbc_command_scale.des_gripper_angles[:] = self._cfg.commander.gripper_angle_scale
        self._wbc_command_range.des_torso_pva[0:2, :] = self._cfg.commander.real_limit.torso_pv_limit.reshape(2, 6) if self._cfg.sim.use_real_robot else self._cfg.commander.real_limit.torso_pv_limit.reshape(2, 6)
        self._wbc_command_range.des_gripper_angles[:] = self._cfg.commander.gripper_angle_range

        self._reset_manipulator_when_switch = self._cfg.commander.reset_manipulator_when_switch

        # for joystick teleoperation
        self._locomotion_height_range = self._cfg.commander.locomotion_height_range
        self._torso_incremental_control = self._cfg.teleoperation.joystick.torso_incremental_control
        self._gripper_task_space_world = self._cfg.teleoperation.joystick.gripper_task_space_world
        #14d (body: xyzrpy, eef_r/l: xyzrpy, grippers: 2 angles)
        self._joystick_command_sub = rospy.Subscriber(self._cfg.teleoperation.joystick.command_topic, Float32MultiArray, self._update_joystick_command_callback, queue_size=1)

        # for human teleoperation
        self._human_time_at_first_stamp = time.time()
        self._human_time_from_first_stamp = 0.
        self._controller_time_at_first_stamp = time.time()
        self._controller_time_from_first_stamp = 0.
        self._time_for_tracking_human = 0.1
        self._reset_from_human = True
        self._first_receive_from_human = True
        self.torso_xyz_threshold = self._cfg.teleoperation.human_teleoperator.torso_xyz_threshold
        self.torso_rpy_threshold = self._cfg.teleoperation.human_teleoperator.torso_rpy_threshold
        self.eef_xyz_threshold = self._cfg.teleoperation.human_teleoperator.eef_xyz_threshold
        self.eef_rpy_threshold = self._cfg.teleoperation.human_teleoperator.eef_rpy_threshold
        self.gripper_angle_threshold = self._cfg.teleoperation.human_teleoperator.gripper_angle_threshold
        self.torso_xyz_scale = self._cfg.teleoperation.human_teleoperator.torso_xyz_scale
        self.torso_rpy_scale = self._cfg.teleoperation.human_teleoperator.torso_rpy_scale
        self.eef_xyz_scale = self._cfg.teleoperation.human_teleoperator.eef_xyz_scale
        self.eef_rpy_scale = self._cfg.teleoperation.human_teleoperator.eef_rpy_scale
        self.gripper_angle_scale = self._cfg.teleoperation.human_teleoperator.gripper_angle_scale
        self.torso_xyz_max_step = self._cfg.teleoperation.human_teleoperator.torso_xyz_max_step
        self.torso_rpy_max_step = self._cfg.teleoperation.human_teleoperator.torso_rpy_max_step
        self.eef_xyz_max_step = self._cfg.teleoperation.human_teleoperator.eef_xyz_max_step
        self.eef_rpy_max_step = self._cfg.teleoperation.human_teleoperator.eef_rpy_max_step
        self.gripper_angle_max_step = self._cfg.teleoperation.human_teleoperator.gripper_angle_max_step
        #20d (body: xyzrpy, eef_r: xyzrpy, eef_l: xyzrpy, grippers: 2 angles)
        self._human_command_sub = rospy.Subscriber(self._cfg.commander.human_command_topic, Float32MultiArray, self._update_human_command_callback, queue_size=1)

    def reset(self):
        self._updating_buffer = False
        self._accessing_buffer = False
        self._command_executed = True
        self._first_receive_from_human = True

        self._wbc_command.reset(keep_mode=True)
        self._wbc_command_buffer.reset()
        self._wbc_command_last_executed.reset()
        self._robot._update_state(reset_estimator=True)

        if self._reset_manipulator_when_switch:
            self._wbc_command.des_gripper_pva[0, 0, 3:6] = self._cfg.manipulator.reset_pos_sim[0:3]
            self._wbc_command.des_gripper_pva[0, 1, 3:6] = self._cfg.manipulator.reset_pos_sim[4:7]
            self._wbc_command.des_gripper_angles[0] = self._cfg.manipulator.reset_pos_sim[3]
            self._wbc_command.des_gripper_angles[1] = self._cfg.manipulator.reset_pos_sim[7]
        else:
            self._wbc_command.des_gripper_pva[0, :, 3:6] = self._robot.joint_pos[self._env_ids, self._robot._manipulator_joint_idx].cpu().numpy().reshape(2, 3)
            self._wbc_command.des_gripper_angles[:] = self._robot.gripper_angles_np[self._env_ids]

    def compute_command_for_wbc(self):
        if self._command_executed:
            self._wbc_command_last_executed = self._wbc_command.copy()
        else:
            self._wbc_command = self._wbc_command_last_executed.copy()
        while self._accessing_buffer:
            pass

    def activate_commander(self):
        self._commander_active = True

    def deactivate_commander(self):
        self._commander_active = False

    def feedback_execution(self, command_executed):
        self._command_executed = command_executed
















