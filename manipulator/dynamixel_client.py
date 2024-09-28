"""Communication using the DynamixelSDK."""
import atexit
import logging
import time
from typing import Optional, Sequence, Union
from manipulator.dynamixel_servo import BaseServo
import numpy as np
import dynamixel_sdk


def dynamixel_cleanup_handler():
    """Cleanup function to ensure Dynamixels are disconnected properly."""
    open_clients = list(DynamixelClient.OPEN_CLIENTS)
    for open_client in open_clients:
        if open_client.port_handler.is_using:
            logging.warning('Forcing client to close.')
        open_client.port_handler.is_using = False
        open_client.disconnect()


def signed_to_unsigned(value: int, size: int) -> int:
    """Converts the given value to its unsigned representation."""
    if value < 0:
        bit_size = 8 * size
        max_value = (1 << bit_size) - 1
        value = max_value + value
    return value


def unsigned_to_signed(value: int, size: int) -> int:
    """Converts the given value from its unsigned representation."""
    bit_size = 8 * size
    if (value & (1 << (bit_size - 1))) != 0:
        value = -((1 << bit_size) - value)
    return value


class DynamixelClient:
    """Client for communicating with Dynamixel motors. This only supports Protocol 2."""

    # The currently open clients.
    OPEN_CLIENTS = set()

    def __init__(self,
                 motor_model: BaseServo,
                 motor_ids: Sequence[int],
                 dof_idx: Sequence[int],
                 gripper_idx: Sequence[int],
                 manipulator_1_idx: Sequence[int],
                 manipulator_2_idx: Sequence[int],
                 port: str = '/dev/ttyUSB1',
                 baudrate: int = 1000000,
                 lazy_connect: bool = False,
                 pos_scale: Optional[float] = None,
                 vel_scale: Optional[float] = None,
                 cur_scale: Optional[float] = None):
        """Initializes a new client.

        Args:
            motor_ids: All motor IDs being used by the client.
            dof_idx: The indices of the motors that are used as the joints of the manipulators.
            gripper_idx: The indices of the motors that are used as the grippers.
            manipulator_1_idx: The indices of the motors that are used as the manipulator_1.
            manipulator_2_idx: The indices of the motors that are used as the manipulator_2.
            port: The Dynamixel device to talk to. e.g.
                - Linux: /dev/ttyUSB0
                - Mac: /dev/tty.usbserial-*
                - Windows: COM1
            baudrate: The Dynamixel baudrate to communicate with.
            lazy_connect: If True, automatically connects when calling a method
                that requires a connection, if not already connected.
            pos_scale: The scaling factor for the positions. This is
                motor-dependent. If not provided, uses the default scale.
            vel_scale: The scaling factor for the velocities. This is
                motor-dependent. If not provided uses the default scale.
            cur_scale: The scaling factor for the currents. This is
                motor-dependent. If not provided uses the default scale.
        """
        self._motor_model = motor_model
        self._cmd_addr = motor_model.address
        self._cmd_scale = motor_model.scale
        self._motor_ids = list(motor_ids)
        self._dof_idx = list(dof_idx)
        self._dof_motor_ids = list(np.array(self._motor_ids)[self._dof_idx])
        self._gripper_idx = list(gripper_idx)
        self._gripper_motor_ids = list(np.array(self._motor_ids)[self._gripper_idx])
        self._manipulator_1_idx = list(manipulator_1_idx)
        self._manipulator_1_motor_ids = list(np.array(self._motor_ids)[self._manipulator_1_idx])
        self._manipulator_2_idx = list(manipulator_2_idx)
        self._manipulator_2_motor_ids = list(np.array(self._motor_ids)[self._manipulator_2_idx])
        self._port_name = port
        self._baudrate = baudrate
        self._lazy_connect = lazy_connect

        self.pos_scale = pos_scale if pos_scale is not None else self._cmd_scale.POSITION_SCALE
        self.vel_scale = vel_scale if vel_scale is not None else self._cmd_scale.VELOCITY_SCALE
        self.current_pos = np.zeros(len(motor_ids))
        self.current_vel = np.zeros(len(motor_ids))

        self.dxl = dynamixel_sdk
        self.port_handler = self.dxl.PortHandler(port)
        self.packet_handler = self.dxl.PacketHandler(self._motor_model.protocol_version)
        self._manipulator_1_pos_vel_reader = self.dxl.GroupSyncRead(self.port_handler, self.packet_handler, self._cmd_addr.PRESENT_POS_VEL[0], self._cmd_addr.PRESENT_POS_VEL[1])
        self._manipulator_2_pos_vel_reader = self.dxl.GroupSyncRead(self.port_handler, self.packet_handler, self._cmd_addr.PRESENT_POS_VEL[0], self._cmd_addr.PRESENT_POS_VEL[1])
        for motor_id in self._motor_ids:
            if motor_id in self._manipulator_1_motor_ids:
                pos_addparam_result = self._manipulator_1_pos_vel_reader.addParam(motor_id)
            elif motor_id in self._manipulator_2_motor_ids:
                pos_addparam_result = self._manipulator_2_pos_vel_reader.addParam(motor_id)
            if not pos_addparam_result:
                print("[ID:%03d] groupSyncRead addparam failed" % motor_id)
                quit()
        self._sync_writers = {}
        self.OPEN_CLIENTS.add(self)

    @property
    def is_connected(self) -> bool:
        return self.port_handler.is_open

    def connect(self):
        assert not self.is_connected, 'Client is already connected.'
        if self.port_handler.openPort():
            logging.info('Succeeded to open port: %s', self._port_name)
        else:
            raise OSError(
                ('Failed to open port at {} (Check that the device is powered '
                 'on and connected to your computer).').format(self._port_name))
        if self.port_handler.setBaudRate(self._baudrate):
            logging.info('Succeeded to set baudrate to %d', self._baudrate)
        else:
            raise OSError(
                ('Failed to set the baudrate to {} (Ensure that the device was '
                 'configured for this baudrate).').format(self._baudrate))

    def disconnect(self):
        if not self.is_connected:
            return
        if self.port_handler.is_using:
            logging.error('Port handler in use; cannot disconnect.')
            return
        # Ensure motors are disabled at the end.
        self.set_torque_enabled(self._motor_ids, False, retries=0)
        self.port_handler.closePort()
        if self in self.OPEN_CLIENTS:
            self.OPEN_CLIENTS.remove(self)

    def read_all_pos_vel(self):
        self.read_manipulator_1_pos_vel()
        self.read_manipulator_2_pos_vel()
        return self.current_pos, self.current_vel

    # def read_all_pos_vel(self):
    #     dxl_comm_result = self._all_pos_reader.txRxPacket()
    #     if dxl_comm_result != COMM_SUCCESS:
    #         print("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))
    #         print("Failed to get present position and velocity of the manipulators!")
    #     else:
    #         for i, motor_id in enumerate(self._motor_ids):
    #             pos_data = self._all_pos_reader.getData(motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
    #             vel_data = self._all_pos_reader.getData(motor_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
    #             self.current_pos[i] = float(unsigned_to_signed(pos_data, LEN_PRESENT_POSITION)) * self.pos_scale
    #             self.current_vel[i] = float(unsigned_to_signed(vel_data, LEN_PRESENT_VELOCITY)) * self.vel_scale
    #     return self.current_pos, self.current_vel


    def read_manipulator_1_pos_vel(self):
        dxl_comm_result = self._manipulator_1_pos_vel_reader.txRxPacket()
        if dxl_comm_result != self._motor_model.command_success:
            print("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))
            print("Failed to get present position and velocity of the manipulator_1!!!!!!!!!!")
        else:
            for i, motor_id in enumerate(self._manipulator_1_motor_ids):
                pos_data = self._manipulator_1_pos_vel_reader.getData(motor_id, self._cmd_addr.PRESENT_POSITION[0], self._cmd_addr.PRESENT_POSITION[1])
                vel_data = self._manipulator_1_pos_vel_reader.getData(motor_id, self._cmd_addr.PRESENT_VELOCITY[0], self._cmd_addr.PRESENT_VELOCITY[1])
                self.current_pos[self._manipulator_1_idx[i]] = float(unsigned_to_signed(pos_data, self._cmd_addr.PRESENT_POSITION[1])) * self.pos_scale
                self.current_vel[self._manipulator_1_idx[i]] = float(unsigned_to_signed(vel_data, self._cmd_addr.PRESENT_VELOCITY[1])) * self.vel_scale

    def read_manipulator_2_pos_vel(self):
        dxl_comm_result = self._manipulator_2_pos_vel_reader.txRxPacket()
        if dxl_comm_result != self._motor_model.command_success:
            print("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))
            print("Failed to get present position and velocity of the manipulator_2!!!!!!!!!!")
        else:
            for i, motor_id in enumerate(self._manipulator_2_motor_ids):
                pos_data = self._manipulator_2_pos_vel_reader.getData(motor_id, self._cmd_addr.PRESENT_POSITION[0], self._cmd_addr.PRESENT_POSITION[1])
                vel_data = self._manipulator_2_pos_vel_reader.getData(motor_id, self._cmd_addr.PRESENT_VELOCITY[0], self._cmd_addr.PRESENT_VELOCITY[1])
                self.current_pos[self._manipulator_2_idx[i]] = float(unsigned_to_signed(pos_data, self._cmd_addr.PRESENT_POSITION[1])) * self.pos_scale
                self.current_vel[self._manipulator_2_idx[i]] = float(unsigned_to_signed(vel_data, self._cmd_addr.PRESENT_VELOCITY[1])) * self.vel_scale

    def set_torque_enabled(self,
                           motor_ids: Sequence[int],
                           enabled: bool,
                           retries: int = -1,
                           retry_interval: float = 0.25):
        """Sets whether torque is enabled for the motors.

        Args:
            motor_ids: The motor IDs to configure.
            enabled: Whether to engage or disengage the motors.
            retries: The number of times to retry. If this is <0, will retry
                forever.
            retry_interval: The number of seconds to wait between retries.
        """
        remaining_ids = list(motor_ids)
        while remaining_ids:
            remaining_ids = self.write_byte(
                remaining_ids,
                int(enabled),
                self._cmd_addr.TORQUE_ENABLE[0],
            )
            if remaining_ids:
                logging.error('Could not set torque %s for IDs: %s',
                              'enabled' if enabled else 'disabled',
                              str(remaining_ids))
            if retries == 0:
                break
            time.sleep(retry_interval)
            retries -= 1

    def write_desired_pos(self, motor_ids: Sequence[int],
                          positions: np.ndarray):
        """Writes the given desired positions.

        Args:
            motor_ids: The motor IDs to write to.
            positions: The joint angles in radians to write.
        """
        assert len(motor_ids) == len(positions)

        # Convert to Dynamixel position space.
        positions = positions / self.pos_scale
        self.sync_write(motor_ids, positions, self._cmd_addr.GOAL_POSITION[0],
                        self._cmd_addr.GOAL_POSITION[1])

    def write_byte(
            self,
            motor_ids: Sequence[int],
            value: int,
            address: int,
    ) -> Sequence[int]:
        """Writes a value to the motors.

        Args:
            motor_ids: The motor IDs to write to.
            value: The value to write to the control table.
            address: The control table address to write to.

        Returns:
            A list of IDs that were unsuccessful.
        """
        self.check_connected()
        errored_ids = []
        for motor_id in motor_ids:
            comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
                self.port_handler, motor_id, address, value)
            success = self.handle_packet_result(
                comm_result, dxl_error, motor_id, context='write_byte')
            if not success:
                errored_ids.append(motor_id)
        return errored_ids

    def sync_write(self, motor_ids: Sequence[int],
                   values: Sequence[Union[int, float]], address: int,
                   size: int):
        """Writes values to a group of motors.

        Args:
            motor_ids: The motor IDs to write to.
            values: The values to write.
            address: The control table address to write to.
            size: The size of the control table value being written to.
        """
        self.check_connected()
        key = (address, size)
        if key not in self._sync_writers:
            self._sync_writers[key] = self.dxl.GroupSyncWrite(
                self.port_handler, self.packet_handler, address, size)
        sync_writer = self._sync_writers[key]

        errored_ids = []
        for motor_id, desired_pos in zip(motor_ids, values):
            value = signed_to_unsigned(int(desired_pos), size=size)
            value = value.to_bytes(size, byteorder='little')
            success = sync_writer.addParam(motor_id, value)
            if not success:
                errored_ids.append(motor_id)

        if errored_ids:
            logging.error('Sync write failed for: %s', str(errored_ids))

        comm_result = sync_writer.txPacket()
        self.handle_packet_result(comm_result, context='sync_write')

        sync_writer.clearParam()

    def check_connected(self):
        """Ensures the robot is connected."""
        if self._lazy_connect and not self.is_connected:
            self.connect()
        if not self.is_connected:
            raise OSError('Must call connect() first.')

    def handle_packet_result(self,
                             comm_result: int,
                             dxl_error: Optional[int] = None,
                             dxl_id: Optional[int] = None,
                             context: Optional[str] = None):
        """Handles the result from a communication request."""
        error_message = None
        if comm_result != self.dxl.COMM_SUCCESS:
            error_message = self.packet_handler.getTxRxResult(comm_result)
        elif dxl_error is not None:
            error_message = self.packet_handler.getRxPacketError(dxl_error)
        if error_message:
            if dxl_id is not None:
                error_message = '[Motor ID: {}] {}'.format(
                    dxl_id, error_message)
            if context is not None:
                error_message = '> {}: {}'.format(context, error_message)
            logging.error(error_message)
            return False
        return True

    def convert_to_unsigned(self, value: int, size: int) -> int:
        """Converts the given value to its unsigned representation."""
        if value < 0:
            max_value = (1 << (8 * size)) - 1
            value = max_value + value
        return value

    def __enter__(self):
        """Enables use as a context manager."""
        if not self.is_connected:
            self.connect()
        return self

    def __exit__(self, *args):
        """Enables use as a context manager."""
        self.disconnect()

    def __del__(self):
        """Automatically disconnect on destruction."""
        self.disconnect()


# Register global cleanup function.
atexit.register(dynamixel_cleanup_handler)

if __name__ == '__main__':
    import argparse
    import itertools

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--motors',
        required=True,
        help='Comma-separated list of motor IDs.')
    parser.add_argument(
        '-d',
        '--device',
        default='/dev/ttyUSB0',
        help='The Dynamixel device to connect to.')
    parser.add_argument(
        '-b', '--baud', default=1000000, help='The baudrate to connect with.')
    parsed_args = parser.parse_args()

    motors = [int(motor) for motor in parsed_args.motors.split(',')]

    way_points = [np.zeros(len(motors)), np.full(len(motors), np.pi)]

    with DynamixelClient(motors, parsed_args.device,
                         parsed_args.baud) as dxl_client:
        for step in itertools.count():
            if step > 0 and step % 50 == 0:
                way_point = way_points[(step // 100) % len(way_points)]
                print('Writing: {}'.format(way_point.tolist()))
                dxl_client.write_desired_pos(motors, way_point)
            read_start = time.time()
            pos_now, vel_now, cur_now = dxl_client.read_pos_vel_cur()
            if step % 5 == 0:
                print('[{}] Frequency: {:.2f} Hz'.format(
                    step, 1.0 / (time.time() - read_start)))
                print('> Pos: {}'.format(pos_now.tolist()))
                print('> Vel: {}'.format(vel_now.tolist()))
                print('> Cur: {}'.format(cur_now.tolist()))
