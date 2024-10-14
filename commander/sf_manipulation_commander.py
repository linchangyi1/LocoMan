from commander.base_commander import BaseCommander
import numpy as np
from fsm.finite_state_machine import FSM_State


class SingleFootManipCommander(BaseCommander):
    def __init__(self, robot, env_ids=0):
        super().__init__(robot, env_ids=env_ids)
        self._wbc_command.operation_mode = FSM_State.SF_MANIPULATION
        self._manipulate_leg_idx = 0

    def reset(self):
        super().reset()
        self._manipulate_leg_idx = self._robot._cur_single_leg_idx.value
        self._wbc_command.contact_state[self._manipulate_leg_idx] = False
        self._wbc_command.des_foot_pva[0, self._manipulate_leg_idx, :] = self._robot.foot_pos_w_np[self._env_ids, self._manipulate_leg_idx]

    def _update_joystick_command_callback(self, command_msg):
        if self._commander_active:
            while self._accessing_buffer:
                pass
            self._updating_buffer = True
            command_np = np.array(command_msg.data)
            self._wbc_command_buffer.des_torso_pva[0, :] = command_np[0:6]
            self._wbc_command_buffer.des_foot_pva[0, self._manipulate_leg_idx, :] = command_np[6:9]
            self._updating_buffer = False
            for _ in range(10):
                pass

    def _update_human_command_callback(self, command_msg):
        pass

    def compute_command_for_wbc(self):
        super().compute_command_for_wbc()
        self._accessing_buffer = True
        self._wbc_command.des_torso_pva[0, :] += self._wbc_command_buffer.des_torso_pva[0, :] * self._wbc_command_scale.des_torso_pva[0, :]
        self._wbc_command.des_torso_pva[0, :] = np.clip(self._wbc_command.des_torso_pva[0, :], -self._wbc_command_range.des_torso_pva[0, :], self._wbc_command_range.des_torso_pva[0, :])
        self._wbc_command.des_foot_pva[0, self._manipulate_leg_idx, :] += self._wbc_command_buffer.des_foot_pva[0, self._manipulate_leg_idx, :] * self._wbc_command_scale.des_foot_pva[0, self._manipulate_leg_idx, :]
        self._wbc_command_buffer.reset(keep_mode=True)
        self._accessing_buffer = False
        return self._wbc_command

