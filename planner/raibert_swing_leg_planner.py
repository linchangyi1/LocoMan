"""The swing leg controller for Raibert-style hopping."""
import numpy as np
import copy
from robot.base_robot import BaseRobot
# from estimator.ground_orientation_estimator import GroundOrientationEstimator
from planner.gait_planner import GaitPlanner
from planner.gait_planner import LegState


def cubic_bezier(x0: np.ndarray, x1: np.ndarray, t: float) -> np.ndarray:
    progress = t**3 + 3 * t**2 * (1 - t)
    return x0 + progress * (x1 - x0)


def _gen_swing_foot_trajectory(input_phase: float, start_pos: np.ndarray,
                            mid_pos: np.ndarray,
                            end_pos: np.ndarray) -> np.ndarray:
    cutoff = 0.46
    # print(f"input_phase, cutoff: {input_phase, cutoff}")
    if input_phase < cutoff:
        t = input_phase / cutoff
        foot_pos = cubic_bezier(start_pos, mid_pos, t)
    else:
        t = (input_phase - cutoff) / (1 - cutoff)
        foot_pos = cubic_bezier(mid_pos, end_pos, t)
    return foot_pos


class RaibertSwingLegPlanner:
    """Controls the swing leg position using Raibert's formula.
    For details, please refer to chapter 2 in "Legged robbots that balance" by
    Marc Raibert. The key idea is to stablize the swing foot's location based on
    the CoM moving speed."""
    def __init__(self,
                robot: BaseRobot,
                env_ids,
                # ground_orientation_estimator: GroundOrientationEstimator,
                gait_generator: GaitPlanner,
                desired_speed=np.array([0.5, 0.]),
                desired_twisting_speed = 0.,
                desired_height = 0.,
                foot_landing_clearance = 0.0,
                foot_height = 0.15):
        """
        Args:
            robot: A robot instance.
            gait_generator: Generates the stance/swing pattern.
            state_estimator: Estiamtes the CoM speeds.
            desired_speed: Behavior parameters. X-Y speed.
            desired_twisting_speed: Behavior control parameters.
            desired_height: Desired standing height.
            foot_landing_clearance: The foot clearance on the ground at the end of
            the swing cycle.
        """
        self._robot = robot
        self._env_ids = env_ids
        # self._ground_estimator = ground_orientation_estimator
        self._gait_generator = gait_generator

        self._last_leg_state = gait_generator.desired_leg_state
        self._desired_speed = np.array((desired_speed[0], desired_speed[1], 0))
        self._desired_twisting_speed = desired_twisting_speed
        self._desired_height = desired_height
        self._desired_landing_height = np.array((0, 0, desired_height - foot_landing_clearance))
        self._foot_height = foot_height

        self._hip_positions_torso_frame = np.array([[0.1881, -0.12675, 0],
                                                   [0.1881, 0.12675, 0],
                                                   [-0.1881, -0.12675, 0],
                                                   [-0.1881, 0.12675, 0]])

        self._phase_switch_foot_local_position = None
        self.reset()

    def reset(self):
        self._last_leg_state = self._gait_generator.desired_leg_state
        # self._phase_switch_foot_local_position = self._ground_estimator.torso_orientation_matrix_zero_yaw.dot(self._robot.foot_pos_b_np[self._env_ids].T).T
        self._phase_switch_foot_local_position = self._robot.foot_pos_w_np[self._env_ids]

    def update(self):
        # Detects phase switch for each leg so we can remember the feet position at the beginning of the swing phase.
        new_leg_state = self._gait_generator.desired_leg_state
        # rot_mat = self._ground_estimator.torso_orientation_matrix_zero_yaw
        for leg_id, state in enumerate(new_leg_state):
            if (state == LegState.SWING and state != self._last_leg_state[leg_id]):
                # self._phase_switch_foot_local_position[leg_id] = rot_mat.dot(self._robot.foot_pos_b_np[self._env_ids][leg_id])
                self._phase_switch_foot_local_position[leg_id] = self._robot.foot_pos_w_np[self._env_ids][leg_id]
        self._last_leg_state = copy.deepcopy(new_leg_state)

    def get_desired_foot_positions(self):
        # ground_normal = self._ground_estimator.ground_normal_vec
        # ground_normal = np.array([0., 0., 1.])
        # rot_mat = self._ground_estimator.torso_orientation_matrix_zero_yaw
        # com_velocity_torso_frame = self._robot.torso_lin_vel_b_np[self._env_ids]
        # com_velocity = rot_mat.dot(com_velocity_torso_frame)
        com_vel_zy = self._robot.torso_lin_vel_w_np[self._env_ids]

        # angular_vel_torso_frame = np.array(self._robot.torso_ang_vel_b_np[self._env_ids])
        # hip_positions = rot_mat.dot(self._hip_positions_torso_frame.T).T
        hip_offset_zy = self._robot.torso_rot_mat_w2b_np[self._env_ids].dot(self._hip_positions_torso_frame.T).T

        foot_positions = []
        for leg_id, leg_state in enumerate(self._gait_generator.leg_state):
            if leg_state in (LegState.STANCE,
                # LegState.EARLY_CONTACT,
                LegState.LOSE_CONTACT,):
                foot_positions.append(np.zeros(3))
                continue

            # hip_offset = hip_positions[leg_id]
            # hip_velocity = com_velocity +rot_mat.dot(np.cross(angular_vel_torso_frame, self._hip_positions_torso_frame[leg_id]))
            hip_velocity = com_vel_zy + self._robot.torso_rot_mat_w2b_np[self._env_ids].dot(np.cross(self._robot.torso_ang_vel_b_np[self._env_ids], self._hip_positions_torso_frame[leg_id]))
            # Mid air position
            mid_x, mid_y, mid_z = hip_offset_zy[leg_id]
            # mid_z += (-(self._desired_height - self._foot_height)) / ground_normal[2]
            mid_z += (-(self._desired_height - self._foot_height))
            mid_position = np.array([mid_x, mid_y, mid_z])

            # Use raibert heuristic to determine target foot position
            target_position = (hip_velocity * self._gait_generator.stance_duration[leg_id] / 2)
            target_position[0] = np.clip(target_position[0], -0.15, 0.15)
            target_position[1] = np.clip(target_position[1], -0.08, 0.08)

            # target_position[2] = (
            #     -self._desired_landing_height[2] -
            #     target_position[0] * ground_normal[0] -
            #     target_position[1] * ground_normal[1]) / ground_normal[2]
            target_position[2] = -self._desired_landing_height[2]
            # print(f"target_position: {target_position}")
            target_position += hip_offset_zy[leg_id]
            # print(f"target_position: {target_position}")
            

            foot_position = _gen_swing_foot_trajectory(self._gait_generator.normalized_phase[leg_id],
                                                       self._phase_switch_foot_local_position[leg_id],
                                                       mid_position,
                                                       target_position)
            foot_positions.append(foot_position)

            # print('\n------------------')
            # print(f"leg_id, foot_position: {leg_id, foot_position}")

        return np.array(foot_positions)


    @property
    def foot_height(self):
        return self._foot_height

    @foot_height.setter
    def foot_height(self, foot_height):
        self._foot_height = foot_height

    @property
    def foot_landing_clearance(self):
        return self._desired_height - self._desired_landing_height[2]

    @foot_landing_clearance.setter
    def foot_landing_clearance(self, landing_clearance):
        self._desired_landing_height = np.array((0., 0., self._desired_height - landing_clearance))

    @property
    def desired_speed(self):
        return self._desired_speed

    @desired_speed.setter
    def desired_speed(self, desired_speed):
        self._desired_speed = desired_speed

    @property
    def desired_twisting_speed(self):
        return self._desired_twisting_speed

    @desired_twisting_speed.setter
    def desired_twisting_speed(self, desired_twisting_speed):
        self._desired_twisting_speed = desired_twisting_speed
















