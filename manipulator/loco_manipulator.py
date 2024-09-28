from config.config import Cfg
import numpy as np
from manipulator.dynamixel_servo import XC330Servo
from manipulator.dynamixel_client import DynamixelClient
import time
import math
import serial.tools.list_ports
import subprocess


class LocoManipulator:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.motor_model = XC330Servo()
        self.motor_ids = cfg.manipulator.motor_ids
        self.gripper_idx = cfg.manipulator.gripper_idx
        self.num_motors = len(self.motor_ids)
        self.cmd_addr = XC330Servo().address

        try:
            self.dxl_client = DynamixelClient(self.motor_model,
                                              self.motor_ids,
                                              cfg.manipulator.dof_idx,
                                              cfg.manipulator.gripper_idx,
                                              cfg.manipulator.manipulator_1_idx,
                                              cfg.manipulator.manipulator_2_idx,
                                              self.get_device_port(),
                                              cfg.manipulator.baudrate)
            self.dxl_client.connect()
            print("Connected to the loco-manipulators")
        except Exception:
            print("Failed to connect to the loco-manipulators")

        self.s2r_scale = cfg.manipulator.s2r_scale
        self.s2r_offset = cfg.manipulator.s2r_offset
        self.kP = cfg.manipulator.kP
        self.kI = cfg.manipulator.kI
        self.kD = cfg.manipulator.kD
        self.CurrLim = np.ones(self.num_motors) * cfg.manipulator.curr_lim
        self.gripper_delta_max = cfg.manipulator.gripper_delta_max
        self.reset_pos_sim = cfg.manipulator.reset_pos_sim
        self.reset_time = cfg.manipulator.reset_time

        self.circle_offset = np.zeros(self.num_motors)
        self.cur_pos_real = np.zeros(self.num_motors)
        self.cur_vel_real = np.zeros(self.num_motors)
        self.cur_pos_vel_sim = np.zeros((2, self.num_motors))
        self.des_pos_real = np.zeros(self.num_motors)

        self.reset()

    def reset(self):
        self.dxl_client.set_torque_enabled(self.motor_ids, False)
        self.set_operating_mode('current_position')
        self.set_PID_current_params()
        self.dxl_client.set_torque_enabled(self.motor_ids, True)
        for _ in range(3):
            self.update_circle_offset()
        self.move_to_target_pos(self.reset_pos_sim, self.reset_time)
        self.update_manipulator_sim_state()

    def move_to_target_pos(self, target_pos_sim, duration=None):
        target_pos_sim = target_pos_sim.clip(-3.3, 3.3)
        target_pos_real = self.circle_offset+target_pos_sim*self.s2r_scale+self.s2r_offset

        if duration is None:
            self.des_pos_real = target_pos_real
            self.des_pos_real[self.gripper_idx] = np.clip(self.des_pos_real[self.gripper_idx], self.cur_pos_real[self.gripper_idx]-self.gripper_delta_max, self.cur_pos_real[self.gripper_idx]+self.gripper_delta_max)
            self.dxl_client.write_desired_pos(self.motor_ids, self.des_pos_real)
        else:
            delta_t = 0.02
            for t in np.arange(delta_t, duration+delta_t, delta_t):
                time_begin = time.time()
                blend_ratio = min(t / duration, 1)
                self.des_pos_real = blend_ratio * target_pos_real + (1 - blend_ratio) * self.cur_pos_real
                self.dxl_client.write_desired_pos(self.motor_ids, self.des_pos_real)
                time_cost = time.time() - time_begin
                time.sleep(max(delta_t - time_cost, 0))


    def update_manipulator_sim_state(self):
        self.cur_pos_real[:], self.cur_vel_real[:] = self.dxl_client.read_all_pos_vel()
        self.cur_pos_vel_sim[0, :] = ((self.cur_pos_real - self.circle_offset - self.s2r_offset) / self.s2r_scale)
        self.cur_pos_vel_sim[1, :] = (self.cur_vel_real / self.s2r_scale)
        return self.cur_pos_vel_sim
    
    def set_operating_mode(self, mode='current_position'):
        operating_mode = 5 if mode == 'current_position' else 3
        self.dxl_client.sync_write(self.motor_ids, np.ones(self.num_motors)*operating_mode, self.cmd_addr.Operating_Mode[0], self.cmd_addr.Operating_Mode[1])

    def set_PID_current_params(self):
        self.dxl_client.sync_write(self.motor_ids, self.kP, self.cmd_addr.Position_P_GAIN[0], self.cmd_addr.Position_P_GAIN[1])
        self.dxl_client.sync_write(self.motor_ids, self.kI, self.cmd_addr.Position_I_GAIN[0], self.cmd_addr.Position_I_GAIN[1])
        self.dxl_client.sync_write(self.motor_ids, self.kD, self.cmd_addr.Position_D_GAIN[0], self.cmd_addr.Position_D_GAIN[1])
        self.dxl_client.sync_write(self.motor_ids, self.CurrLim, self.cmd_addr.GOAL_CURRENT[0], self.cmd_addr.GOAL_CURRENT[1])

    def update_circle_offset(self):
        # when the operating mode is set to position control mode, the read position will be reset to 0~2pi
        self.cur_pos_real[:], self.cur_vel_real[:] = self.dxl_client.read_all_pos_vel()
        if np.sum(self.cur_pos_real) < 1e-4:
            print('Failed to read the position')
            quit()
        else:
            self.circle_offset[(self.cur_pos_real / math.pi * 180)>180] = 2*math.pi


    def get_device_port(self):
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            if port.vid is not None and port.pid is not None:
                device_vid_pid = "{:04x}:{:04x}".format(port.vid, port.pid)
                if device_vid_pid.lower() == self.cfg.manipulator.vid_pid.lower():
                    device_name = port.device.split('/')[-1]
                    set_success = set_latency_timer(device_name, 1)
                    if set_success:
                        return port.device
        print("Device not found")
        return None

def set_latency_timer(device, latency):
    # Building the path to the latency_timer file for the device
    latency_file_path = f"/sys/bus/usb-serial/devices/{device}/latency_timer"
    # Command to change the latency timer
    cmd = f"echo {latency} | sudo tee {latency_file_path}"
    try:
        # Executing the command
        subprocess.run(cmd, shell=True, check=True)
        print(f"Latency timer set to {latency} for {device}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to set latency timer for {device}: {e}")
        return False



def main():
    loco_manipualtor = LocoManipulator(Cfg)

    gripper_actions = np.array([[math.pi, -0.05, 0., 0.02, math.pi, 0.05, 0., 0.02],
                                    [math.pi, -0.05, 0., 0.02, math.pi/2, 1.05, 0.5, 0.02],
                                    [math.pi, -0.05, 0., 0.02, math.pi, 0.05, 0., 0.02],
                                    [math.pi, -0.05, 0., 0.02, math.pi/2, 1.05, -0.5, 0.02],
                                    [math.pi, -0.05, 0., 0.02, math.pi, 0.05, 0., 0.02],
                                    ])

    while True:
        print(loco_manipualtor.update_manipulator_sim_state())
        time.sleep(0.02)


if __name__ == "__main__":
    main()
