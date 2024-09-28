"""Simple state estimator for Go1 robot."""
import numpy as np
from filterpy.kalman import KalmanFilter
from estimator.moving_window_filter import MovingWindowFilter
from utilities.orientation_utils_numpy import rpy_to_rot_mat
import time
import torch

_DEFAULT_WINDOW_SIZE = 1
_ANGULAR_VELOCITY_FILTER_WINDOW_SIZE = 1


def convert_to_skew_symmetric(x: np.ndarray) -> np.ndarray:
  return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


class KFStateEstimator:
  """Estimates base velocity of A1 robot.
  The velocity estimator consists of a state estimator for CoM velocity.
  Two sources of information are used:
  The integrated reading of accelerometer and the velocity estimation from
  contact legs. The readings are fused together using a Kalman Filter.
  """
  def __init__(self,
               robot,
               accelerometer_variance: np.ndarray = np.array(
                   [1.42072319e-05, 1.57958752e-05, 8.75317619e-05]),
               sensor_variance: np.ndarray = np.array(
                   [0.33705298, 0.14858707, 0.68439632]) * 0.03,
               initial_variance: float = 0.1,
               use_external_contact_estimator: bool = False):
    """Initiates the velocity/height estimator.
    See filterpy documentation in the link below for more details.
    https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
    Args:
      robot: the robot class for velocity estimation.
      accelerometer_variance: noise estimation for accelerometer reading.
      sensor_variance: noise estimation for motor velocity reading.
      initial_covariance: covariance estimation of initial state.
    """
    self._robot = robot
    self._use_external_contact_estimator = use_external_contact_estimator
    self._foot_contact = np.ones(4)

    self.filter = KalmanFilter(dim_x=3, dim_z=3, dim_u=3)
    self.filter.x = np.zeros(3)
    self._initial_variance = initial_variance
    self._accelerometer_variance = accelerometer_variance
    self._sensor_variance = sensor_variance
    self.filter.P = np.eye(3) * self._initial_variance  # State covariance
    self.filter.Q = np.eye(3) * accelerometer_variance
    self.filter.R = np.eye(3) * sensor_variance

    self.filter.H = np.eye(3)  # measurement function (y=H*x)
    self.filter.F = np.eye(3)  # state transition matrix
    self.filter.B = np.eye(3)

    self.ma_filter = MovingWindowFilter(window_size=_DEFAULT_WINDOW_SIZE)
    self._angular_velocity_filter = MovingWindowFilter(
        window_size=_ANGULAR_VELOCITY_FILTER_WINDOW_SIZE)
    self._angular_velocity = np.zeros(3)
    self._estimated_velocity = np.zeros(3)
    self._estimated_position = np.array([0., 0., self._robot._cfg.locomotion.desired_pose[2]])

    # buffers for updating the robot state
    self._zeroyaw_rpy_world = np.zeros(3)
    self._zeroyaw_rot_mat_zy2w = np.eye(3)
    self._torso_rpy_w2b = np.zeros(3)


    self.reset()

  def reset(self, env_ids=None):
    self.filter.x = np.zeros(3)
    self.filter.P = np.eye(3) * self._initial_variance
    self._last_timestamp = time.time()
    self._delta_time_s = .0
    self._last_torso_velocity_sim = np.zeros(3)
    self._estimated_velocity = self.filter.x.copy()

  def _get_velocity_and_height_observation(self):
    # torso_orientation = self._robot.torso_orientation_quat
    # rot_mat = self._robot.pybullet_client.getMatrixFromQuaternion(
    #     torso_orientation)
    # rot_mat = np.array(rot_mat).reshape((3, 3))
    R_globalw2b = rpy_to_rot_mat(np.array(self._robot._raw_state.imu.rpy))
    # foot_positions = self._robot.foot_positions_in_torso_frame
    foot_positions = self._robot.foot_pos_b_np[0]
    foot_velocities = self._robot.foot_vel_b_np[0]
    # ang_vel_cross = convert_to_skew_symmetric(self._robot.base_angular_velocity_torso_frame)
    ang_vel_cross = convert_to_skew_symmetric(self._angular_velocity)
    observed_velocities, observed_heights = [], []
    if self._use_external_contact_estimator:
      foot_contact = self._foot_contact.copy()
    else:
      foot_contact = self._robot.foot_contact_np[0]
    if self._robot._log_info_now:
      print('foot_contact: ', foot_contact)
    
    for leg_id in range(4):
      if foot_contact[leg_id]:
        # jacobian = self._robot.compute_foot_jacobian(leg_id)
        # # Only pick the jacobian related to joint motors
        # joint_velocities = self._robot.motor_velocities[leg_id *
        #                                                3:(leg_id + 1) * 3]
        # leg_velocity_in_torso_frame = jacobian.dot(joint_velocities)[:3]
        observed_velocities.append(
            -R_globalw2b.dot(foot_velocities[leg_id] + ang_vel_cross.dot(foot_positions[leg_id])))
        observed_heights.append(-R_globalw2b.dot(foot_positions[leg_id])[2])
    return observed_velocities, observed_heights

  def update_foot_contact(self, foot_contact):
    self._foot_contact = foot_contact.copy()

  def update(self):
    """Propagate current state estimate with new accelerometer reading."""
    self._delta_time_s = self._robot._dt if self._delta_time_s == .0 else (time.time() - self._last_timestamp)
    self._last_timestamp = time.time()
    R_globalw2b = rpy_to_rot_mat(np.array(self._robot._raw_state.imu.rpy))
    calibrated_acc = R_globalw2b.dot(np.array(self._robot._raw_state.imu.accelerometer)) + np.array([0., 0., -9.8])
    self.filter.predict(u=calibrated_acc * self._delta_time_s)

    (observed_velocities,  observed_heights) = self._get_velocity_and_height_observation()

    if observed_velocities:
      observed_velocities = np.mean(observed_velocities, axis=0)
      # multiplier = np.clip(
      #     1 + (np.sqrt(observed_velocities[0]**2 + \
      #     observed_velocities[1]**2) -
      #          0.3), 1, 1.3)
      # observed_velocities[0] *= 1.3
      self.filter.update(observed_velocities)

    self._estimated_velocity = self.ma_filter.calculate_average(self.filter.x.copy())
    self._angular_velocity = self._angular_velocity_filter.calculate_average(np.array(self._robot._raw_state.imu.gyroscope))

    self._estimated_position += self._delta_time_s * self._estimated_velocity
    if observed_heights:
      self._estimated_position[2] = np.mean(observed_heights)
    if self._robot._log_info_now:
      print('delta_time_s: ', self._delta_time_s)
      print('calibrated_acc: ', calibrated_acc)
      print('observed_velocities: ', observed_velocities)
      print('observed_heights: ', observed_heights)
      print('estimated_velocity: ', self._estimated_velocity)
      print('angular_velocity: ', self._angular_velocity)
      print('estimated_position: ', self._estimated_position)


  def set_robot_torso_state(self):
    # self._robot._torso_pos_w[:] = torch.tensor(self._estimated_position, dtype=torch.float, device=self._robot._device)
    self._robot._torso_pos_w[:, 2] = self._estimated_position[2]
    self._torso_rpy_w2b[0] = self._robot._raw_state.imu.rpy[0]
    self._torso_rpy_w2b[1] = self._robot._raw_state.imu.rpy[1]
    self._robot._torso_rpy_w2b[:] = torch.tensor(self._torso_rpy_w2b, dtype=torch.float, device=self._robot._device)
    self._robot._torso_rot_mat_w2b[:] = torch.tensor(rpy_to_rot_mat(self._torso_rpy_w2b), dtype=torch.float, device=self._robot._device)
    self._robot._torso_rot_mat_b2w[:] = self._robot._torso_rot_mat_w2b.transpose(-2, -1)

    self._zeroyaw_rpy_world[2] = self._robot._raw_state.imu.rpy[2]
    self._zeroyaw_rot_mat_zy2w[:] = rpy_to_rot_mat(self._zeroyaw_rpy_world).T
    self._robot._torso_lin_vel_w[:] = torch.tensor(self._zeroyaw_rot_mat_zy2w.dot(self._estimated_velocity.reshape(-1, 1)).flatten(), dtype=torch.float, device=self._robot._device)
    self._robot._torso_lin_vel_b[:] = torch.matmul(self._robot._torso_rot_mat_b2w, self._robot._torso_lin_vel_w.reshape(-1, 1)).flatten()
    self._robot._torso_ang_vel_w[:] = torch.tensor(self._zeroyaw_rot_mat_zy2w.dot(self._angular_velocity.reshape(-1, 1)).flatten(), dtype=torch.float, device=self._robot._device)
    self._robot._torso_ang_vel_b[:] = torch.matmul(self._robot._torso_rot_mat_b2w, self._robot._torso_ang_vel_w.reshape(-1, 1)).flatten()


  @property
  def estimated_velocity(self):
    return self._estimated_velocity.copy()

  @property
  def estimated_position(self):
    return self._estimated_position.copy()

  @property
  def angular_velocity(self):
    return self._angular_velocity
