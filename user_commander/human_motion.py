import rospy
from std_msgs.msg import Float32MultiArray, Int32
import socket
import numpy as np
from config.config import Cfg
from utilities.orientation_utils_numpy import rot_mat_to_rpy


class HumanTeleoperator():
    def __init__(self):
        rospy.init_node('human_teleoperation')
        self.command_publisher = rospy.Publisher(Cfg.commander.human_command_topic, Float32MultiArray, queue_size=1)

        # command buffer
        self.command = np.zeros(20)  # body: xyzrpy, eef_r: xyzrpy, eef_l: xyzrpy, grippers: 2 angles
        self.command_msg = Float32MultiArray()
        self.command_msg.data = self.command.tolist()

        # initial values
        self.initial_receive = True
        self.init_torso_pos = np.zeros(3)
        self.init_torso_rot = np.eye(3)
        self.init_eef_pos = np.zeros((2, 3))
        self.init_eef_rot = np.array([np.eye(3), np.eye(3)])
        self.init_gripper_angles = np.zeros(2)

        # server-client setup
        self.server_host = Cfg.teleoperation.human_teleoperator.SERVER_HOST
        self.server_port = Cfg.teleoperation.human_teleoperator.SERVER_PORT
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.server_host, self.server_port))
        self.server_socket.listen(1)
        print(f"Listening for incoming connections on {self.server_host}:{self.server_port}")
        self.data_byte_array = bytearray()
        self.data_np_array = np.array([])
        self.start_signal = np.array([-1000.0], dtype=np.float32).tobytes()
        self.end_signal = np.array([-10000.0], dtype=np.float32).tobytes()

        self.is_changing_reveive_status = False
        self.begin_to_receive = False
        self.reset_signal_subscriber = rospy.Subscriber(Cfg.teleoperation.human_teleoperator.receive_action_topic, Int32, self.reset_signal_callback, queue_size=1)
        self.teleoperation_mode = Cfg.teleoperation.human_teleoperator.mode
        self.is_changing_teleop_mode = False
        self.teleop_mode_subscriber = rospy.Subscriber(Cfg.teleoperation.human_teleoperator.mode_updata_topic, Int32, self.teleop_mode_callback, queue_size=1)
        self.rate = rospy.Rate(100)


    def reset_signal_callback(self, msg):
        self.is_changing_reveive_status = True
        if msg.data == 1:
            self.begin_to_receive = True
            self.initial_receive = True
            print("Begin to receive. Initial receive status reset.")
        elif msg.data == 0:
            self.begin_to_receive = False
            self.initial_receive = True
            print("No longer receiving. Initial receive status reset.")
        self.is_changing_reveive_status = False

    def teleop_mode_callback(self, msg):
        self.is_changing_teleop_mode = True
        self.teleoperation_mode = msg.data
        print(f"Teleoperation mode updated to {self.teleoperation_mode}.")
        self.is_changing_teleop_mode = False

    def run(self):
        while not rospy.is_shutdown():
            self.client_socket, self.client_address = self.server_socket.accept()
            print(f"Connection from {self.client_address} has been established.")
            self.client_socket.settimeout(5.0)
            try:
                self.data_byte_array = bytearray()
                while not rospy.is_shutdown():
                    if self.is_changing_reveive_status or self.is_changing_teleop_mode:
                        continue
                    try:
                        chunk = self.client_socket.recv(224)  # one float32 is 4 bytes, actually 56 bytes are enough
                        if not chunk:
                            self.client_socket.close()
                            print("Connection closed.")
                            break
                        self.data_byte_array.extend(chunk)
                        # Check for the presence of start and end signals
                        start_index = self.data_byte_array.find(self.start_signal)
                        end_index = self.data_byte_array.find(self.end_signal, start_index + len(self.start_signal)) if start_index != -1 else -1

                        while start_index != -1 and end_index != -1 and end_index > start_index:                            
                            # Extract the data between start and end signals
                            data_to_process = self.data_byte_array[start_index + len(self.start_signal):end_index]
                            if data_to_process:
                                self.data_np_array = np.frombuffer(data_to_process, dtype=np.float32)

                                self.command[:] = 0
                                # Process data_np_array as required...
                                if self.initial_receive and self.begin_to_receive:
                                    print("Received array:", self.data_np_array)
                                    if self.teleoperation_mode != 3:
                                        self.init_torso_pos[:] = self.data_np_array[22:25].copy()
                                        self.init_torso_rot[:] = self.data_np_array[13:22].reshape((3, 3)).copy()
                                        if self.teleoperation_mode != 0:
                                            eef_idx = int(self.teleoperation_mode - 1)
                                            self.init_eef_pos[eef_idx] = self.data_np_array[9:12].copy()
                                            self.init_eef_rot[eef_idx] = self.data_np_array[0:9].reshape((3, 3)).copy()
                                            self.init_gripper_angles[eef_idx] = self.data_np_array[12]
                                    elif self.teleoperation_mode == 3:
                                        print('Teleoperation mode 3.')
                                        for i in range(2):
                                            self.init_eef_pos[i, :] = self.data_np_array[i*13+9:i*13+12].copy()
                                            self.init_eef_rot[i, :] = self.data_np_array[i*13:i*13+9].reshape((3, 3)).copy()
                                            self.init_gripper_angles[i] = self.data_np_array[i*13+12]
                                    print("Initial values set.")
                                    print("init_eef_pos:", self.init_eef_pos)
                                    print("init_eef_rot:", self.init_eef_rot)
                                    print("init_gripper_angles:", self.init_gripper_angles)
                                elif self.begin_to_receive:
                                    if self.teleoperation_mode != 3:
                                        print("Received array:", self.data_np_array)
                                        self.command[0:3] = self.data_np_array[22:25] - self.init_torso_pos
                                        self.command[3:6] = rot_mat_to_rpy(self.data_np_array[13:22].reshape((3, 3)) @ self.init_torso_rot.T)
                                        if self.teleoperation_mode != 0:
                                            eef_idx = int(self.teleoperation_mode - 1)
                                            self.command[eef_idx*6+6:eef_idx*6+9] = self.data_np_array[9:12] - self.init_eef_pos[eef_idx]
                                            self.command[eef_idx*6+9:eef_idx*6+12] = rot_mat_to_rpy(self.data_np_array[eef_idx*13:eef_idx*13+9].reshape((3, 3)) @ self.init_eef_rot[eef_idx].T)
                                            self.command[eef_idx+18] = self.data_np_array[12] - self.init_gripper_angles[eef_idx]
                                    elif self.teleoperation_mode == 3:
                                        print('\n Teleoperation mode 3.')
                                        for i in range(2):
                                            self.command[i*6+6:i*6+9] = self.data_np_array[i*13+9:i*13+12] - self.init_eef_pos[i]
                                            self.command[i*6+9:i*6+12] = rot_mat_to_rpy(self.data_np_array[i*13:i*13+9].reshape((3, 3)) @ self.init_eef_rot[i].T)
                                            self.command[i+18] = self.data_np_array[i*13+12] - self.init_gripper_angles[i]
                                        print("R_eef_xyz:", self.command[6:9])
                                        print("R_eef_rpy:", self.command[9:12])
                                        print("L_eef_xyz:", self.command[12:15])
                                        print("L_eef_rpy:", self.command[15:18])
                                        print("R_gripper_angle:", self.command[18])
                                        print("L_gripper_angle:", self.command[19])


                                self.command_msg.data = self.command.tolist()
                                self.command_publisher.publish(self.command_msg)
                                if self.initial_receive:
                                    print("Published command:", self.command_msg.data)
                                self.initial_receive = False

                            # Remove processed segment from the buffer
                            self.data_byte_array = self.data_byte_array[end_index + len(self.end_signal):]

                            # Look for next segment within the remaining buffer
                            start_index = self.data_byte_array.find(self.start_signal)
                            end_index = self.data_byte_array.find(self.end_signal, start_index + len(self.start_signal)) if start_index != -1 else -1

                        self.rate.sleep()
                    except socket.timeout:
                        print("Socket timed out. No data received.")
                        continue  # Continue listening for data
                    except Exception as e:
                        print(f"Error receiving data: {e}")
            except KeyboardInterrupt:
                break  # Exit the loop on Ctrl+C
            finally:
                self.client_socket.close()
                print("Connection closed.")
        quit()



if __name__ == "__main__":
    human_teleoperator = HumanTeleoperator()
    try:
        human_teleoperator.run()
    except rospy.ROSInterruptException:
        pass


