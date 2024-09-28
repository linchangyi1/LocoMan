from mode_commander.base_commander import BaseCommander
import numpy as np
from planner.gait_planner import GaitPlanner, LegState
from planner.raibert_swing_leg_planner import RaibertSwingLegPlanner
from fsm.finite_state_machine import FSM_State


class LocomotionCommander(BaseCommander):
    def __init__(self, robot, env_ids=0):
        super().__init__(robot, env_ids=0)
        self._wbc_command.operation_mode = FSM_State.LOCOMOTION
        self._gait_generator = GaitPlanner(robot, env_ids)
        self._swing_leg_controller = RaibertSwingLegPlanner(robot, env_ids, self._gait_generator, foot_landing_clearance=self._cfg.locomotion.foot_landing_clearance_real if self._cfg.sim.use_real_robot else self._cfg.locomotion.foot_landing_clearance_sim)
        self._going_to_stand = False
        self._switching_time = self._cfg.switcher.locomotion_switching_time
        self._switching_steps = 0

    def reset(self):
        super().reset()
        self._gait_generator.reset()
        self._swing_leg_controller.reset()

        self._wbc_command.des_torso_pva[0, :] = self._cfg.locomotion.desired_pose
        self._wbc_command.des_torso_pva[1, :] = self._cfg.locomotion.desired_velocity
        torso_height = self._robot.torso_pos_w_np[self._env_ids, 2]  # don't use average foot height that is "foot_radius" smaller than the real torso height
        self._swing_leg_controller._foot_height = (torso_height / self._wbc_command.des_torso_pva[0, 2]) * self._cfg.locomotion.foot_height
        self._wbc_command.des_torso_pva[0, 2] = torso_height

    def _update_joystick_command_callback(self, command_msg):
        if self._commander_active:
            while self._accessing_buffer:
                pass
            self._updating_buffer = True
            command_np = np.array(command_msg.data)
            self._wbc_command_buffer.des_torso_pva[0, 2:5] = command_np[2:5]  # z, roll, pitch
            self._wbc_command_buffer.des_torso_pva[1, 0:2] = command_np[0:2]  # vx, vy
            self._wbc_command_buffer.des_torso_pva[1, 5] = command_np[5]  # wz
            self._updating_buffer = False
            for _ in range(10):
                pass

    def _update_human_command_callback(self, command_msg):
        pass

    def compute_command_for_wbc(self):
        super().compute_command_for_wbc()
        self._gait_generator.update()
        self._swing_leg_controller.update()
        self._wbc_command.contact_state[:] = np.array([state in (LegState.STANCE, LegState.EARLY_CONTACT, LegState.LOSE_CONTACT) for state in self._gait_generator.leg_state])
        self._wbc_command.des_foot_pva[0, :, :] = self._swing_leg_controller.get_desired_foot_positions()

        self._accessing_buffer = True
        # roll, pitch, vx, vy, wz
        if self._torso_incremental_control:
            self._wbc_command.des_torso_pva[0, 3:5] += self._wbc_command_buffer.des_torso_pva[0, 3:5] * self._wbc_command_scale.des_torso_pva[0, 3:5]
            self._wbc_command.des_torso_pva[0, 3:5] = np.clip(self._wbc_command.des_torso_pva[0, 3:5], -self._wbc_command_range.des_torso_pva[0, 3:5], self._wbc_command_range.des_torso_pva[0, 3:5])
            self._wbc_command.des_torso_pva[1, :] += self._wbc_command_buffer.des_torso_pva[1, :] * self._wbc_command_scale.des_torso_pva[1, :]
            self._wbc_command.des_torso_pva[1, :] = np.clip(self._wbc_command.des_torso_pva[1, :], -self._wbc_command_range.des_torso_pva[1, :], self._wbc_command_range.des_torso_pva[1, :])
        else:
            self._wbc_command.des_torso_pva[0, 3:5] = self._wbc_command_buffer.des_torso_pva[0, 3:5] * self._wbc_command_range.des_torso_pva[0, 3:5]
            self._wbc_command.des_torso_pva[1, :] = self._wbc_command_buffer.des_torso_pva[1, :] * self._wbc_command_range.des_torso_pva[1, :]
        # z(height) always uses delta command
        self._wbc_command.des_torso_pva[0, 2] += self._wbc_command_buffer.des_torso_pva[0, 2] * self._wbc_command_scale.des_torso_pva[0, 2]
        self._wbc_command.des_torso_pva[0, 2] = np.clip(self._wbc_command.des_torso_pva[0, 2], self._locomotion_height_range[0], self._locomotion_height_range[1])

        self._wbc_command_buffer.reset(keep_mode=True)
        self._accessing_buffer = False

        if self._going_to_stand:
            height = self._wbc_command.des_torso_pva[0, 2]
            self._wbc_command.des_torso_pva[:] = 0
            self._wbc_command.des_torso_pva[0, 2] = height
            self._switching_steps += 1

        return self._wbc_command


    def prepare_to_stand(self):
        self._going_to_stand = True

    def check_finished(self):
        if self._switching_steps * self._robot._dt > self._switching_time and np.sum(self._wbc_command.contact_state) == 4:
            self._going_to_stand = False
            self._switching_steps = 0
            return True
        else:
            return False







