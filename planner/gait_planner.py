import enum
import numpy as np
from robot.base_robot import BaseRobot


class LegState(enum.Enum):
  """The state of a leg during locomotion."""
  SWING = 0
  STANCE = 1
  # A swing leg that collides with the ground.
  EARLY_CONTACT = 2
  # A stance leg that loses contact.
  LOSE_CONTACT = 3


class GaitPlanner:
    def __init__(self, robot: BaseRobot, env_ids):
        self._robot = robot
        self._env_ids = env_ids
        self._early_touchdown_phase_threshold = self._robot._cfg.locomotion.early_touchdown_phase_threshold
        self._lose_contact_phase_threshold = self._robot._cfg.locomotion.lose_contact_phase_threshold

        self.gait_params = self._robot._cfg.locomotion.gait_params
        self.current_phase = np.zeros(4)
        self.prev_frame_robot_time = 0
        self.swing_cutoff = 2 * np.pi * self.gait_params[4]
        self.desired_leg_state = np.array([LegState.STANCE for _ in range(4)])
        self.leg_state = np.array([LegState.STANCE for _ in range(4)])
        self.normalized_phase = np.zeros(4)

    def reset(self):
        self.current_phase = np.zeros(4)
        self.prev_frame_robot_time = self._robot.time_since_reset[self._env_ids]
        self.swing_cutoff = 2 * np.pi * self.gait_params[4]

    def update(self):
        # Calculate the amount of time passed
        current_robot_time = self._robot.time_since_reset[self._env_ids]
        frame_duration = self._robot.time_since_reset[self._env_ids] - self.prev_frame_robot_time
        self.prev_frame_robot_time = current_robot_time
        # Propagate phase for front-right leg
        self.current_phase[0] += 2 * np.pi * frame_duration * self.gait_params[0]
        # Offset for remaining legs
        self.current_phase[1:4] = self.current_phase[0] + self.gait_params[1:4]
        self.swing_cutoff = 2 * np.pi * (1 - self.gait_params[4])
        self.stance_duration = 1 / self.gait_params[0] * (
            1 - self.gait_params[4]) * np.ones(4)

        modulated_phase = np.mod(self.current_phase + 2 * np.pi, 2 * np.pi)
        self.desired_leg_state = np.array([LegState.SWING if phase > self.swing_cutoff else LegState.STANCE for phase in modulated_phase])
        self.normalized_phase = np.where(modulated_phase < self.swing_cutoff, modulated_phase / self.swing_cutoff, (modulated_phase - self.swing_cutoff) / (2 * np.pi - self.swing_cutoff))

        self.leg_state = self.desired_leg_state.copy()
        # contact_state = self._robot.foot_contact_np[self._env_ids]
        # for leg_id in range(self._robot._num_legs):
        #     if (self.leg_state[leg_id] == LegState.STANCE and not contact_state[leg_id] and self.normalized_phase[leg_id] > self._lose_contact_phase_threshold):
        #         self.leg_state[leg_id] = LegState.LOSE_CONTACT
        #     elif (self.leg_state[leg_id] == LegState.SWING and contact_state[leg_id] and self.normalized_phase[leg_id] > self._early_touchdown_phase_threshold):
        #         self.leg_state[leg_id] = LegState.EARLY_CONTACT
        


    # @property
    # def leg_state(self):
    #     leg_state = self.desired_leg_state.copy()
    #     contact_state = self._robot.foot_contact_np[self._env_ids]

    #     for leg_id in range(self._robot._num_legs):
    #         if (leg_state[leg_id] == LegState.STANCE and not contact_state[leg_id] and self.normalized_phase[leg_id] > self._lose_contact_phase_threshold):
    #             # logging.info("lost contact detected.")
    #             leg_state[leg_id] = LegState.LOSE_CONTACT
    #         if (leg_state[leg_id] == LegState.SWING and contact_state[leg_id] and self.normalized_phase[leg_id] > self._early_touchdown_phase_threshold):
    #             # logging.info("early touch down detected.")
    #             leg_state[leg_id] = LegState.EARLY_CONTACT
    #     return leg_state








