from config.config import Cfg
from robot.base_robot import BaseRobot
from robot.motors import MotorCommand, MotorControlMode
from utilities.rotation_utils import rpy_vel_to_skew_synmetric_mat
import torch
import time
import platform
system_arch = platform.machine()
if system_arch == 'x86_64':
    import unitree_legged_sdk.lib.python.amd64.robot_interface as sdk
elif system_arch == 'aarch64':
    import unitree_legged_sdk.lib.python.arm64.robot_interface as sdk
else:
    raise ImportError("Unsupported architecture: {}".format(system_arch))
import rospy
from sensor_msgs.msg import JointState


class RealRobot(BaseRobot):
    def __init__(self, cfg: Cfg):
        super().__init__(cfg)
        self._init_interface()
        self._init_buffers()

    def _init_interface(self):
        if self._use_gripper:
            self.des_pos_sim_pub = rospy.Publisher(self._cfg.manipulator.des_pos_sim_topic, JointState, queue_size=1)
            self.des_pos_sim_msg = JointState()
            self.des_pos_sim_msg.name = ['do_not_update_state']

            self.cur_state_sim_sub = rospy.Subscriber(self._cfg.manipulator.cur_state_sim_topic, JointState, self.update_gripper_state_callback)

            manipulator_motor_number = len(self._cfg.manipulator.motor_ids)
            self._gripper_joint_state_buffer = torch.zeros((2, manipulator_motor_number), dtype=torch.float, device=self._device, requires_grad=False)
            self._received_first_gripper_state = False

            self._gripper_reset_pos = torch.tensor(self._cfg.manipulator.reset_pos_sim, dtype=torch.float, device=self._device, requires_grad=False)
            self._des_gripper_pos = torch.zeros_like(self._gripper_reset_pos)
            self._dof_within_manipulator_idx = self._cfg.manipulator.dof_idx
            self._gripper_within_manipulator_idx = self._cfg.manipulator.gripper_idx
            # self._close_gripper = True
            self._open_degree = 0.5
            self._close_degree = self._gripper_reset_pos[3]
            self._update_gripper_interval = 1.0 / self._cfg.manipulator.update_manipulator_freq
            self._last_time_gripper_updated = time.time()
            self._requesting_gripper_state = False

        self._udp = sdk.UDP(0xff, 8080, "192.168.123.10", 8007)
        self._safe = sdk.Safety(sdk.LeggedType.Go1)
        self._power_protect_level = self._cfg.motor_control.power_protect_level
        self._cmd = sdk.LowCmd()
        self._raw_state = sdk.LowState()
        self._udp.InitCmdData(self._cmd)


    def _init_buffers(self):
        self._last_update_state_time = time.time()
        self._contact_force_threshold = torch.zeros(self._num_legs, dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_init_pos = self._motors.init_positions

    def update_gripper_state_callback(self, joint_msg: JointState):
        if not self._received_first_gripper_state:
            self._received_first_gripper_state = True
        self._last_time_gripper_updated = time.time()
        self._gripper_joint_state_buffer[0, :] = torch.tensor(joint_msg.position, dtype=torch.float, device=self._device, requires_grad=False)
        self._gripper_joint_state_buffer[1, :] = torch.tensor(joint_msg.velocity, dtype=torch.float, device=self._device, requires_grad=False)
        self._requesting_gripper_state = False

    def reset(self, reset_time: float = 2.5):
        # make sure the communication is ready
        zero_action = MotorCommand(desired_position=torch.zeros((self._num_envs, self._num_joints), device=self._device),
                            kp=torch.zeros((self._num_envs, self._num_joints), device=self._device),
                            desired_velocity=torch.zeros((self._num_envs, self._num_joints), device=self._device),
                            kd=torch.zeros((self._num_envs, self._num_joints), device=self._device),
                            desired_extra_torque=torch.zeros((self._num_envs, self._num_joints), device=self._device))
        
        if self._use_gripper:
            rospy.sleep(1.0)
            self._last_time_gripper_updated = time.time()
            self._requesting_gripper_state = True
            self.des_pos_sim_msg.name = ['update_state']
            self.des_pos_sim_msg.header.stamp = rospy.Time.now()
            self.des_pos_sim_msg.position = list(self._gripper_reset_pos.cpu().numpy())
            self.des_pos_sim_pub.publish(self.des_pos_sim_msg)
            zero_action.desired_position[:, self._manipulator_joint_idx] = self._gripper_reset_pos[self._dof_within_manipulator_idx]

        for _ in range(10):
            self.step(zero_action, gripper_cmd=True)
        print("Ready to reset the robot!")
        
        initial_joint_pos = self.joint_pos
        stable_joint_pos = self._motors.init_positions
        # Stand up in 1 second, and collect the foot contact forces afterwords
        reset_time = self._cfg.motor_control.reset_time + 1.0
        standup_time = self._cfg.motor_control.reset_time
        stand_foot_forces = []
        for t in torch.arange(0, reset_time, self._dt):
            blend_ratio = min(t / standup_time, 1)
            desired_joint_pos = blend_ratio * stable_joint_pos + (1 - blend_ratio) * initial_joint_pos
            stand_up_action = MotorCommand(desired_position=desired_joint_pos,
                                kp=self._motors.kps,
                                desired_velocity=torch.zeros((self._num_envs, self._num_joints), device=self._device),
                                kd=self._motors.kds,
                                desired_extra_torque=torch.zeros((self._num_envs, self._num_joints), device=self._device))
            
            if not rospy.is_shutdown():
                self.step(stand_up_action, MotorControlMode.POSITION, gripper_cmd=False)

            if t > standup_time:
                stand_foot_forces.append(self.raw_contact_force)
        # Calibrate foot force sensors
        if stand_foot_forces:
            stand_foot_forces_tensor = torch.stack(stand_foot_forces)
            mean_foot_forces = torch.mean(stand_foot_forces_tensor, dim=0)
        else:
            mean_foot_forces = torch.zeros_like(self._contact_force_threshold)
        self._contact_force_threshold[:] = mean_foot_forces * 0.8
        self._update_state(reset_estimator=True, env_ids=torch.arange(self._num_envs, device=self._device))  # for updating foot contact state
        self._num_step[:] = 0
        print("Robot reset done!")


    def step(self, action: MotorCommand, motor_control_mode: MotorControlMode = None, gripper_cmd=True):
        self._num_step[:] += 1
        self._log_info_now = self._log_info and self._num_step[0] % self._log_interval == 0
        self._apply_action(action, motor_control_mode, gripper_cmd=gripper_cmd)
        time.sleep(max(self._dt- (time.time()-self._last_update_state_time), 0))
        self._update_state()

    def _apply_action(self, action: MotorCommand, motor_control_mode: MotorControlMode = None, gripper_cmd=True):
        if motor_control_mode is None:
            motor_control_mode = self._motors._motor_control_mode
        if motor_control_mode == MotorControlMode.POSITION:
            for motor_id in range(self._dog_num_joints):
                self._cmd.motorCmd[motor_id].q = action.desired_position.cpu().numpy()[0, self._dog_joint_idx[motor_id]]
                self._cmd.motorCmd[motor_id].Kp = action.kp.cpu().numpy()[0, self._dog_joint_idx[motor_id]]
                self._cmd.motorCmd[motor_id].dq = 0.
                self._cmd.motorCmd[motor_id].Kd = action.kd.cpu().numpy()[0, self._dog_joint_idx[motor_id]]
                self._cmd.motorCmd[motor_id].tau = 0.
        elif motor_control_mode == MotorControlMode.TORQUE:
            for motor_id in range(self._dog_num_joints):
                self._cmd.motorCmd[motor_id].q = 0.
                self._cmd.motorCmd[motor_id].Kp = 0.
                self._cmd.motorCmd[motor_id].dq = 0.
                self._cmd.motorCmd[motor_id].Kd = 0.
                self._cmd.motorCmd[motor_id].tau = action.desired_torque.cpu().numpy()[0, self._dog_joint_idx[motor_id]]
        elif motor_control_mode == MotorControlMode.HYBRID:
            for motor_id in range(self._dog_num_joints):
                self._cmd.motorCmd[motor_id].q = action.desired_position.cpu().numpy()[0, self._dog_joint_idx[motor_id]]
                self._cmd.motorCmd[motor_id].Kp = action.kp.cpu().numpy()[0, self._dog_joint_idx[motor_id]]
                self._cmd.motorCmd[motor_id].dq = action.desired_velocity.cpu().numpy()[0, self._dog_joint_idx[motor_id]]
                self._cmd.motorCmd[motor_id].Kd = action.kd.cpu().numpy()[0, self._dog_joint_idx[motor_id]]
                self._cmd.motorCmd[motor_id].tau = action.desired_extra_torque.cpu().numpy()[0, self._dog_joint_idx[motor_id]]
        else:
            raise ValueError('Unknown motor control mode for Go1 robot: {}.'.format(motor_control_mode))
        self._safe.PowerProtect(self._cmd, self._raw_state, self._power_protect_level)
        self._udp.SetSend(self._cmd)
        self._udp.Send()

        if gripper_cmd and self._use_gripper and not self._requesting_gripper_state:
            des_dof_pos = action.desired_position[0, self._manipulator_joint_idx]
            self._des_gripper_pos[0:3] = des_dof_pos[0:3]
            self._des_gripper_pos[4:7] = des_dof_pos[3:6]
            self._des_gripper_pos[[3, 7]] = self._gripper_desired_angles[0, :]

            if (time.time()-self._last_time_gripper_updated) > self._update_gripper_interval:
                self.des_pos_sim_msg.name = ['update_state']
                self._requesting_gripper_state = True
            else:
                self.des_pos_sim_msg.name = ['do_not_update_state']
            self.des_pos_sim_msg.header.stamp = rospy.Time.now()
            self.des_pos_sim_msg.position = list(self._des_gripper_pos.cpu().numpy())
            self.des_pos_sim_pub.publish(self.des_pos_sim_msg)

    def _update_sensors(self):
        self._last_update_state_time = time.time()

        # gripper
        if self._use_gripper:
            if not self._received_first_gripper_state:
                self._joint_pos[:, self._manipulator_joint_idx] = self._gripper_reset_pos[self._dof_within_manipulator_idx]
                self._joint_vel[:, self._manipulator_joint_idx] = 0.0
                self._gripper_angles[:] = self._gripper_reset_pos[self._gripper_within_manipulator_idx]
            else:
                self._joint_pos[:, self._manipulator_joint_idx] = self._gripper_joint_state_buffer[0, self._dof_within_manipulator_idx]
                self._joint_vel[:, self._manipulator_joint_idx] = self._gripper_joint_state_buffer[1, self._dof_within_manipulator_idx]
                self._gripper_angles[:] = self._gripper_joint_state_buffer[0, self._gripper_within_manipulator_idx]

        # dog
        self._udp.Recv()
        self._udp.GetRecv(self._raw_state)
        for motor_id in range(self._dog_num_joints):
            self._joint_pos[:, self._dog_joint_idx[motor_id]] = self._raw_state.motorState[motor_id].q
            self._joint_vel[:, self._dog_joint_idx[motor_id]] = self._raw_state.motorState[motor_id].dq

    def _update_foot_contact_state(self):
        self._foot_contact[:] = self.raw_contact_force > self._contact_force_threshold

    def _update_foot_jocabian_position_velocity(self):
        self._compute_foot_jacobian()

        self._foot_pos_hip[:], self._foot_vel_hip[:] = self._forward_kinematics(return_frame='hip')
        self._foot_pos_b[:] = self._foot_pos_hip + self._HIP_OFFSETS
        self._foot_vel_b[:] = self._foot_vel_hip

    def _update_foot_global_state(self):
        # ----------------- compute foot global position -----------------
        self._foot_pos_w[:] = torch.bmm(self._torso_rot_mat_w2b, self._foot_pos_b.transpose(1, 2)).transpose(1, 2) + self._torso_pos_w.unsqueeze(1)

        # ----------------- compute foot global velocity -----------------
        # Vf^w = Vb^w + [w_b^w] * R_w^b * pf^b + R^w2b * Vf^b
        self._foot_vel_w[:] = self._torso_lin_vel_w.unsqueeze(1) + \
                            torch.bmm(rpy_vel_to_skew_synmetric_mat(self._torso_ang_vel_w), torch.bmm(self._torso_rot_mat_b2w.transpose(-2, -1), self._foot_pos_b.transpose(-2, -1))).transpose(1, 2) +\
                            torch.bmm(self._torso_rot_mat_w2b, self._foot_vel_b.transpose(1, 2)).transpose(1, 2)

    @property
    def raw_contact_force(self):
        return torch.tensor(self._raw_state.footForce, dtype=torch.float, device=self._device, requires_grad=False)


