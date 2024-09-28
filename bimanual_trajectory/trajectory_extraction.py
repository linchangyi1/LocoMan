#!/usr/bin/python

import sys
import time
import numpy as np
import pickle
import platform
system_arch = platform.machine()
if system_arch == 'x86_64':
    import unitree_legged_sdk.lib.python.amd64.robot_interface as sdk
elif system_arch == 'aarch64':
    import unitree_legged_sdk.lib.python.arm64.robot_interface as sdk
else:
    raise ImportError("Unsupported architecture: {}".format(system_arch))
from config.config import Cfg


if __name__ == '__main__':

    HIGHLEVEL = 0xee
    LOWLEVEL  = 0xff

    udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)

    cmd = sdk.HighCmd()
    state = sdk.HighState()
    states = []
    udp.InitCmdData(cmd)

    step_time = Cfg.motor_control.dt / Cfg.bimanual_trajectory.recording_fps_mul_motor_control_fps

    motiontime = 0
    start_time = time.time()
    while time.time() - start_time < 10:
        time_start = time.time()
        udp.Recv()
        udp.GetRecv(state)

        states.append(dict(
            timestamp=time.time() - start_time,
            torso_height=np.array(state.bodyHeight),
            torso_position=np.array(state.position),
            torso_orientation=np.array(state.imu.rpy),
            velocity=np.array(state.velocity),
            angular_velocity=np.array(state.imu.gyroscope),
            motor_position=np.array([m.q for m in state.motorState]),
            motor_vel=np.array([m.dq for m in state.motorState]),
            motor_torque=np.array([m.tauEst for m in state.motorState]),
            motor_acc=np.array([m.ddq for m in state.motorState]),
            foot_force=np.array(state.footForce),
            foot_force_est=np.array(state.footForceEst),
        ))
        udp.Send()

        motiontime += 1
        time.sleep(max(0, step_time - (time.time() - time_start)))

    with open(Cfg.bimanual_trajectory.trajectory_path, "wb") as f:
        pickle.dump(states, f)


