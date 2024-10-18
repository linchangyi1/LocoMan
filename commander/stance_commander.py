from commander.base_commander import BaseCommander
import numpy as np


class StanceCommander(BaseCommander):
    def __init__(self, robot, env_ids=0):
        super().__init__(robot, env_ids=env_ids)

    def reset(self):
        super().reset()

    def _update_joystick_command_callback(self, command_msg):
        if self._commander_active:
            while self._accessing_buffer:
                pass
            self._updating_buffer = True
            self._wbc_command_buffer.des_torso_pva[0, :] = np.array(command_msg.data)[0:6]
            self._updating_buffer = False
            for _ in range(10):
                pass
            
    def _update_human_command_callback(self, command_msg):
        pass

    def compute_command_for_wbc(self):
        super().compute_command_for_wbc()
        self._accessing_buffer = True
        if self._torso_incremental_control:
            self._wbc_command.des_torso_pva[0, :] += self._wbc_command_buffer.des_torso_pva[0, :] * self._wbc_command_scale.des_torso_pva[0, :]
            self._wbc_command.des_torso_pva[0, :] = np.clip(self._wbc_command.des_torso_pva[0, :], -self._wbc_command_range.des_torso_pva[0, :], self._wbc_command_range.des_torso_pva[0, :])
        else:
            self._wbc_command.des_torso_pva[0, :] = self._wbc_command_buffer.des_torso_pva[0, :] * self._wbc_command_range.des_torso_pva[0, :]
        self._wbc_command_buffer.reset(keep_mode=True)
        self._accessing_buffer = False
        return self._wbc_command

    def check_finished(self):
        return True




