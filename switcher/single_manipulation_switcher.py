from fsm.finite_state_machine import FSM_State, SingleLegIndex
from robot.base_robot import BaseRobot
from planner.switch_planner import SwitchPlanner
import numpy as np


class SingleManipSwitcher:
    def __init__(self, robot: BaseRobot, env_ids=0):
        self._robot = robot
        self._device = self._robot._device
        self._env_ids = env_ids
        self._cfg = self._robot._cfg

        # There are two steps from stance to single-arm manipulation (including single-foot and single-gripper):
        # 1. move the base (direction depends on left and right)
        # 2. move the foot and loco-manipulator(foot position and servo angles depends on the legged-arm state)
        self._torso_right_movement = self._cfg.switcher.stance_and_single_manipulation.torso_movement.copy()
        self._torso_left_movement = self._cfg.switcher.stance_and_single_manipulation.torso_movement.copy()
        self._torso_left_movement[1] *= -1.0
        self._foot_movement = self._cfg.switcher.stance_and_single_manipulation.foot_movement.copy()
        self._manipulator_angles = self._cfg.switcher.stance_and_single_manipulation.manipulator_angles.copy()
        self._manipulate_leg_idx = 0
        self._no_manipulate_leg_idx = 0 if self._manipulate_leg_idx == 1 else 1
        self._stance_to_manipulation_trajectory = self._cfg.switcher.stance_and_single_manipulation.stance_to_manipulation_trajectory
        self._manipulation_to_stance_trajectory = self._cfg.switcher.stance_and_single_manipulation.manipulation_to_stance_trajectory

    def stance_to_single_manipulation(self, fsm_state_buffer, single_leg_idx_buffer):
        self._robot._update_state(reset_estimator=True)  # to keep the current base pose
        stance_to_manipulation_trajectory = self._stance_to_manipulation_trajectory.copy()

        # check the direction of the base movement
        if single_leg_idx_buffer == SingleLegIndex.LEFT:
            stance_to_manipulation_trajectory[0, 0:6] = self._torso_left_movement
            self._manipulate_leg_idx = 1
            self._no_manipulate_leg_idx = 0
        else:
            stance_to_manipulation_trajectory[0, 0:6] = self._torso_right_movement
            self._manipulate_leg_idx = 0
            self._no_manipulate_leg_idx = 1

        # then check the foot position and manipulator angles
        foot_pos_w_before_torso_movement = self._robot.foot_pos_w_np[self._env_ids, self._manipulate_leg_idx].copy()
        foot_pos_w_after_torso_movement = foot_pos_w_before_torso_movement - stance_to_manipulation_trajectory[0, 0:3]
        manipulation_init_foot_pos_w = foot_pos_w_after_torso_movement + self._foot_movement
        stance_to_manipulation_trajectory[1, 6:9] = manipulation_init_foot_pos_w
        if fsm_state_buffer == FSM_State.SG_MANIPULATION:
            stance_to_manipulation_trajectory[1, 9+3*self._manipulate_leg_idx:12+3*self._manipulate_leg_idx] = self._manipulator_angles
        else:
            stance_to_manipulation_trajectory[1, 21] *=0.5 # reduce the time if only need to move the foot

        return SwitchPlanner(self._robot, stance_to_manipulation_trajectory, self._manipulate_leg_idx)

    def single_manipulation_to_stance(self):
        manipulation_to_stance_trajectory = self._manipulation_to_stance_trajectory.copy()
        manipulation_to_stance_trajectory[0, 0:3] = self._robot.torso_pos_w_np[self._env_ids]
        manipulation_to_stance_trajectory[0, 3:6] = self._robot.torso_rpy_w2b_np[self._env_ids]

        self._manipulate_leg_idx = 0 if self._robot._cur_single_leg_idx == SingleLegIndex.RIGHT else 1
        self._no_manipulate_leg_idx = 0 if self._manipulate_leg_idx==1 else 1

        foot_pos_w = self._robot.foot_pos_w_np[self._env_ids].copy()
        manipulation_to_stance_trajectory[0, 6:9] = foot_pos_w[self._manipulate_leg_idx]
        manipulation_to_stance_trajectory[1, 6:9] = foot_pos_w[self._no_manipulate_leg_idx] + (foot_pos_w[2+self._manipulate_leg_idx] - foot_pos_w[2+self._no_manipulate_leg_idx])
        foot_pos_w[self._manipulate_leg_idx] = manipulation_to_stance_trajectory[1, 6:9]
        manipulation_to_stance_trajectory[2, 0:2] = np.mean(foot_pos_w[:, 0:2], axis=0)

        if not self._robot._cfg.commander.reset_manipulator_when_switch:
            manipulation_to_stance_trajectory[:, 19+self._manipulate_leg_idx] = self._robot.gripper_angles_np[self._env_ids, self._manipulate_leg_idx]

        if self._robot._cur_fsm_state == FSM_State.SF_MANIPULATION:
            manipulation_to_stance_trajectory[0, -2] = 0.1  # reduce the time for retreating the gripper

        return SwitchPlanner(self._robot, manipulation_to_stance_trajectory, self._manipulate_leg_idx)

